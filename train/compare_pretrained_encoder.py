"""Bounded frozen Whisper encoder comparison; research-only, not the runtime model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from seamtranslation.evaluation import semantic_generation_metrics
from seamtranslation.models import ModelConfig
from seamtranslation.models.semantic import SemanticDecoder
from seamtranslation.tokenization import HindiTokenizer
from seamtranslation.training import WordProjectSemanticDataset
from train.check_audio_dependence import audio_permutation
from train.train_semantic import (
    _generated_content_tokens, _model_config, deterministic_batch_indices,
    scheduled_learning_rate,
)


class CachedSpeechHead(nn.Module):
    def __init__(self, config: ModelConfig, feature_dim: int) -> None:
        super().__init__()
        self.config = config
        self.adapter = nn.Sequential(nn.Linear(feature_dim, config.model_dim),
                                     nn.LayerNorm(config.model_dim))
        self.decoder = SemanticDecoder(
            vocab_size=config.semantic_vocab_size, dim=config.model_dim,
            layers=config.semantic_layers, heads=config.semantic_heads,
            ffn_dim=config.semantic_ffn_dim, max_tokens=config.max_semantic_tokens,
            dropout=config.dropout, pad_token_id=config.pad_token_id,
        )

    def forward(self, features, padding, tokens):
        return self.decoder(tokens, self.adapter(features), padding)

    def generate(self, features, padding):
        return self.decoder.generate(self.adapter(features), padding,
                                     bos_token_id=self.config.bos_token_id,
                                     eos_token_id=self.config.eos_token_id)


def memory_batch(features: list[torch.Tensor], indices: list[int], device: str):
    rows = [features[index] for index in indices]
    lengths = torch.tensor([row.shape[0] for row in rows], device=device)
    memory = nn.utils.rnn.pad_sequence(rows, batch_first=True).to(device, dtype=torch.float32)
    padding = torch.arange(memory.shape[1], device=device)[None, :] >= lengths[:, None]
    return memory, padding


def target_batch(texts, indices, tokenizer, config, device):
    pieces = [tokenizer.encode(texts[index])[:config.max_semantic_tokens - 1]
              for index in indices]
    inputs = [torch.tensor([config.bos_token_id, *row]) for row in pieces]
    labels = [torch.tensor([*row, config.eos_token_id]) for row in pieces]
    return tuple(nn.utils.rnn.pad_sequence(rows, batch_first=True,
                                          padding_value=config.pad_token_id).to(device)
                 for rows in (inputs, labels))


def cached_features(args, condition, manifest_hash):
    from transformers import WhisperFeatureExtractor, WhisperModel
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    cache_path = args.output / f"{condition}_features.pt"
    revision = json.loads((args.encoder / "revision.json").read_text())
    provenance = {"manifest_sha256": manifest_hash, "encoder": revision,
                  "condition": condition, "random_encoder_seed": args.seed,
                  "weights_sha256": sha256((args.encoder / "model.safetensors").read_bytes()).hexdigest()}
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False, map_location="cpu")
        if cache["provenance"] != provenance:
            raise ValueError("cached feature provenance differs")
        return cache
    extractor = WhisperFeatureExtractor.from_pretrained(args.encoder, local_files_only=True)
    pretrained = WhisperModel.from_pretrained(args.encoder, local_files_only=True)
    if condition == "pretrained":
        encoder = pretrained.get_encoder()
    else:
        torch.manual_seed(args.seed)
        encoder = WhisperEncoder(pretrained.config)
    del pretrained
    encoder.to(args.device).eval().requires_grad_(False)
    cache = {"provenance": provenance,
             "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
             "feature_dim": encoder.config.d_model}
    started = time.perf_counter()
    for split in ("train", "validation"):
        dataset = WordProjectSemanticDataset(args.manifest, split=split)
        features, texts, ids = [], [], []
        for index in range(len(dataset)):
            example = dataset[index]
            if example.waveform.numel() > 30 * 16000:
                raise ValueError("audio exceeds Whisper's 30-second window; no silent truncation")
            inputs = extractor(example.waveform.numpy(), sampling_rate=16000,
                               return_tensors="pt").input_features.to(args.device)
            with torch.no_grad():
                hidden = encoder(inputs).last_hidden_state[0]
                length = min(hidden.shape[0], (example.waveform.numel() + 319) // 320)
                features.append(hidden[:length].to("cpu", dtype=torch.float16).clone())
            texts.append(example.target_text)
            ids.append(dataset.rows[index].get("id", index))
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"{condition} cache {split}: {index + 1}/{len(dataset)}", flush=True)
        cache[split] = {"features": features, "texts": texts, "ids": ids}
    cache["elapsed_seconds"] = time.perf_counter() - started
    torch.save(cache, cache_path)
    del encoder
    if args.device == "mps":
        torch.mps.empty_cache()
    return cache


@torch.inference_mode()
def evaluate(head, cache, tokenizer, device, *, count=None, shuffled=False):
    head.eval()
    count = min(count or len(cache["texts"]), len(cache["texts"]))
    indices = audio_permutation(count, 17) if shuffled else list(range(count))
    hypotheses, tokens, eos_flags = [], [], []
    for index in indices:
        features, padding = memory_batch(cache["features"], [index], device)
        row = head.generate(features, padding)[0].cpu().tolist()
        content, ended = _generated_content_tokens(row, tokenizer)
        hypotheses.append(tokenizer.decode(content))
        tokens.append(content)
        eos_flags.append(ended)
    references = cache["texts"][:count]
    return {"metrics": semantic_generation_metrics(hypotheses, references, tokens, eos_flags),
            "hypotheses": hypotheses, "references": references,
            "audio_indices": indices}


def train_condition(args, condition, cache, tokenizer, baseline):
    config = _model_config(baseline["raw_config"], tokenizer)
    torch.manual_seed(args.seed)
    head = CachedSpeechHead(config, cache["feature_dim"]).to(args.device)
    decoder_state = {key.removeprefix("semantic_decoder."): value
                     for key, value in baseline["model_state_dict"].items()
                     if key.startswith("semantic_decoder.")}
    head.decoder.load_state_dict(decoder_state)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0002)
    training = {"learning_rate": 0.0002, "max_steps": args.steps,
                "lr_schedule": {"warmup_steps": min(50, args.steps // 10),
                                "minimum_ratio": 0.1}}
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        head.train()
        indices = deterministic_batch_indices(step - 1, dataset_size=len(cache["train"]["texts"]),
                                              batch_size=9, seed=args.seed)
        features, padding = memory_batch(cache["train"]["features"], indices, args.device)
        inputs, labels = target_batch(cache["train"]["texts"], indices, tokenizer,
                                      config, args.device)
        lr = scheduled_learning_rate(step, training)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, padding, inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                               ignore_index=config.pad_token_id)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            metrics = evaluate(head, cache["validation"], tokenizer, args.device,
                               count=32)["metrics"]
            history.append({"step": step, "loss": loss.item(), "lr": lr, **metrics})
            print(f"{condition} step={step} loss={loss.item():.3f} "
                  f"dev_chrf2={metrics['chrf2']:.3f}", flush=True)
    checkpoint = {"head_state_dict": head.state_dict(), "model_config": asdict(config),
                  "feature_dim": cache["feature_dim"], "provenance": cache["provenance"],
                  "steps": args.steps, "seed": args.seed,
                  "tokenizer_sha256": baseline["tokenizer_sha256"]}
    torch.save(checkpoint, args.output / f"{condition}_head.pt")
    print(f"{condition}: evaluating all validation examples", flush=True)
    correct = evaluate(head, cache["validation"], tokenizer, args.device)
    shuffled = evaluate(head, cache["validation"], tokenizer, args.device, shuffled=True)
    report = {"condition": condition, "steps": args.steps, "seed": args.seed,
              "device": args.device, "provenance": cache["provenance"],
              "baseline_sha256": sha256(args.baseline.read_bytes()).hexdigest(),
              "tokenizer_sha256": baseline["tokenizer_sha256"],
              "torch_version": torch.__version__,
              "train_examples": len(cache["train"]["texts"]),
              "validation_examples": len(cache["validation"]["texts"]),
              "trainable_parameters": sum(p.numel() for p in head.parameters()),
              "frozen_encoder_parameters": cache["encoder_parameters"],
              "elapsed_seconds": time.perf_counter() - started,
              "generation_history": history, "correct_audio": correct,
              "shuffled_audio": shuffled,
              "correct_minus_shuffled_chrf2": correct["metrics"]["chrf2"] -
                                              shuffled["metrics"]["chrf2"]}
    (args.output / f"{condition}_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"condition": condition, **correct["metrics"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("encoder", "manifest", "tokenizer", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if args.steps < 10:
        parser.error("--steps must be at least 10")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = HindiTokenizer(args.tokenizer)
    baseline = torch.load(args.baseline, weights_only=False, map_location="cpu")
    if sha256(args.tokenizer.read_bytes()).hexdigest() != baseline["tokenizer_sha256"]:
        raise ValueError("baseline tokenizer differs")
    manifest_hash = sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_hash != baseline["manifest_sha256"]:
        raise ValueError("baseline manifest differs")
    reports = {}
    for condition in ("random", "pretrained"):
        if (args.output / f"{condition}_results.json").exists():
            report = json.loads((args.output / f"{condition}_results.json").read_text())
            if (report["steps"] != args.steps or report["seed"] != args.seed or
                report["provenance"]["manifest_sha256"] != manifest_hash or
                report["baseline_sha256"] != sha256(args.baseline.read_bytes()).hexdigest()):
                raise ValueError("completed experiment configuration differs")
            reports[condition] = report
            continue
        cache = cached_features(args, condition, manifest_hash)
        reports[condition] = train_condition(args, condition, cache, tokenizer, baseline)
        del cache
    summary = {condition: {"metrics": report["correct_audio"]["metrics"],
                           "correct_minus_shuffled_chrf2": report["correct_minus_shuffled_chrf2"]}
               for condition, report in reports.items()}
    summary["pretrained_minus_random_chrf2"] = (
        reports["pretrained"]["correct_audio"]["metrics"]["chrf2"] -
        reports["random"]["correct_audio"]["metrics"]["chrf2"])
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
