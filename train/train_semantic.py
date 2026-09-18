"""Train, validate, checkpoint, and resume the speech-to-semantic stage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import time

import torch

from seamtranslation.models import ModelConfig, SpeechSemanticModel
from seamtranslation.evaluation import semantic_generation_metrics
from seamtranslation.runtime import detect_runtime
from seamtranslation.tokenization import HindiTokenizer
from seamtranslation.training import (
    WordProjectSemanticDataset,
    collate_semantic_batch,
    semantic_loss,
)


def train(
    config_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    output: Path,
    *,
    resume_path: Path | None = None,
) -> dict:
    if output.exists() and resume_path is None:
        raise FileExistsError(f"output already exists: {output}")
    raw_config = json.loads(config_path.read_text())
    training = raw_config["training"]
    if training.get("precision", "fp32") != "fp32":
        raise ValueError("only fp32 training is currently verified")
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = HindiTokenizer(tokenizer_path)
    model_config = _model_config(raw_config, tokenizer)
    device_name = _device_name(str(training.get("device", "auto")))
    device = torch.device(device_name)
    manifest_sha256 = _sha256_file(manifest_path)
    tokenizer_sha256 = _sha256_file(tokenizer_path)
    train_dataset = WordProjectSemanticDataset(
        manifest_path, split="train", max_examples=int(training["max_examples"])
    )
    evaluation_split = _evaluation_split(training)
    validation_dataset = WordProjectSemanticDataset(
        manifest_path,
        split=evaluation_split,
        max_examples=int(training["max_validation_examples"]),
    )
    model = SpeechSemanticModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
    start_step = 0
    best_validation_loss = float("inf")
    train_history: list[dict[str, float | int]] = []
    validation_history: list[dict[str, float | int]] = []
    generation_history: list[dict[str, object]] = []
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        _validate_resume(
            checkpoint,
            raw_config=raw_config,
            manifest_sha256=manifest_sha256,
            tokenizer_sha256=tokenizer_sha256,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        train_history = list(checkpoint["train_history"])
        validation_history = list(checkpoint["validation_history"])
        generation_history = list(checkpoint.get("generation_history", []))

    output.mkdir(parents=True, exist_ok=resume_path is not None)
    max_steps = int(training["max_steps"])
    if start_step >= max_steps:
        raise ValueError(f"checkpoint step {start_step} is not below max_steps {max_steps}")
    batch_size = int(training["batch_size"])
    evaluation_interval = int(training["evaluation_interval"])
    checkpoint_interval = int(training["checkpoint_interval"])
    log_interval = int(training["log_interval"])
    generation_interval = int(training.get("generation_interval", 0))
    max_generation_examples = int(training.get("max_generation_examples", 0))
    if (generation_interval == 0) != (max_generation_examples == 0):
        raise ValueError(
            "generation_interval and max_generation_examples must both be zero or positive"
        )
    _synchronize(device)
    started = time.perf_counter()

    for step_index in range(start_step, max_steps):
        step = step_index + 1
        model.train()
        indices = deterministic_batch_indices(
            step_index, dataset_size=len(train_dataset), batch_size=batch_size, seed=seed
        )
        batch = collate_semantic_batch(
            [train_dataset[index] for index in indices],
            tokenizer,
            max_tokens=model_config.max_semantic_tokens,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = semantic_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["max_gradient_norm"])
        )
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        train_history.append(
            {"step": step, "loss": loss_value, "gradient_norm": float(gradient_norm)}
        )
        if step == 1 or step % log_interval == 0:
            print(f"step={step} train_loss={loss_value:.6f}")

        should_evaluate = step % evaluation_interval == 0 or step == max_steps
        if should_evaluate:
            validation_loss = evaluate(
                model,
                validation_dataset,
                tokenizer,
                device=device,
                batch_size=batch_size,
                max_tokens=model_config.max_semantic_tokens,
            )
            validation_history.append({"step": step, "loss": validation_loss})
            print(f"step={step} validation_loss={validation_loss:.6f}")
            if generation_interval and (
                step % generation_interval == 0 or step == max_steps
            ):
                generation = evaluate_generation(
                    model,
                    validation_dataset,
                    tokenizer,
                    device=device,
                    batch_size=batch_size,
                    max_examples=max_generation_examples,
                )
                generation_record = {"step": step, **generation}
                generation_history.append(generation_record)
                print(
                    f"step={step} generation_chrf2={generation['chrf2']:.3f} "
                    f"bleu={generation['bleu']:.3f} eos_rate={generation['eos_rate']:.3f} "
                    f"repeated_token_rate={generation['repeated_token_rate']:.3f}"
                )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                _save_checkpoint(
                    output / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_validation_loss=best_validation_loss,
                    train_history=train_history,
                    validation_history=validation_history,
                    generation_history=generation_history,
                    raw_config=raw_config,
                    manifest_sha256=manifest_sha256,
                    tokenizer_sha256=tokenizer_sha256,
                )

        if step % checkpoint_interval == 0 or step == max_steps:
            for checkpoint_path in (
                output / f"checkpoint-step-{step:06d}.pt",
                output / "checkpoint.pt",
            ):
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_validation_loss=best_validation_loss,
                    train_history=train_history,
                    validation_history=validation_history,
                    generation_history=generation_history,
                    raw_config=raw_config,
                    manifest_sha256=manifest_sha256,
                    tokenizer_sha256=tokenizer_sha256,
                )

    _synchronize(device)
    elapsed = time.perf_counter() - started
    metadata = {
        "git_commit": _git_commit(),
        "config": raw_config,
        "model_config": asdict(model_config),
        "manifest_sha256": manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "device": device_name,
        "start_step": start_step,
        "steps": max_steps,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "evaluation_split": evaluation_split,
        "best_validation_loss": best_validation_loss,
        "train_history": train_history,
        "validation_history": validation_history,
        "generation_history": generation_history,
        "elapsed_seconds": elapsed,
        "parameters_trained": model.parameter_count(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


@torch.inference_mode()
def evaluate(
    model: SpeechSemanticModel,
    dataset: WordProjectSemanticDataset,
    tokenizer: HindiTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    max_tokens: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = collate_semantic_batch(
            [dataset[index] for index in range(start, stop)], tokenizer, max_tokens=max_tokens
        ).to(device)
        loss = semantic_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite validation loss")
        losses.append(float(loss.cpu()))
    return sum(losses) / len(losses)


@torch.inference_mode()
def evaluate_generation(
    model: SpeechSemanticModel,
    dataset: WordProjectSemanticDataset,
    tokenizer: HindiTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    max_examples: int,
) -> dict[str, object]:
    """Greedily decode a fixed validation subset and score actual translations."""
    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []
    generated_ids: list[list[int]] = []
    ended_with_eos: list[bool] = []
    sample_outputs: list[dict[str, str]] = []
    stop = min(len(dataset), max_examples)
    for start in range(0, stop, batch_size):
        examples = [dataset[index] for index in range(start, min(start + batch_size, stop))]
        batch = collate_semantic_batch(
            examples, tokenizer, max_tokens=model.config.max_semantic_tokens
        ).to(device)
        rows = model.generate(batch.waveforms, batch.waveform_lengths).cpu().tolist()
        for example, row in zip(examples, rows, strict=True):
            tokens, ended = _generated_content_tokens(row, tokenizer)
            hypothesis = tokenizer.decode(tokens)
            hypotheses.append(hypothesis)
            references.append(example.target_text)
            generated_ids.append(tokens)
            ended_with_eos.append(ended)
            if len(sample_outputs) < 3:
                sample_outputs.append(
                    {"reference": example.target_text, "hypothesis": hypothesis}
                )

    metrics = semantic_generation_metrics(
        hypotheses, references, generated_ids, ended_with_eos
    )
    return {**metrics, "samples": sample_outputs}


def _generated_content_tokens(
    row: list[int], tokenizer: HindiTokenizer
) -> tuple[list[int], bool]:
    tokens: list[int] = []
    ended = False
    ignored = {tokenizer.special_ids.bos, tokenizer.special_ids.pad}
    for token in row:
        if token == tokenizer.special_ids.eos:
            ended = True
            break
        if token not in ignored:
            tokens.append(token)
    return tokens, ended


def deterministic_batch_indices(
    step_index: int, *, dataset_size: int, batch_size: int, seed: int
) -> list[int]:
    indices: list[int] = []
    for offset in range(batch_size):
        sample_index = step_index * batch_size + offset
        epoch, position = divmod(sample_index, dataset_size)
        order = list(range(dataset_size))
        random.Random(seed + epoch).shuffle(order)
        indices.append(order[position])
    return indices


def _evaluation_split(training: dict) -> str:
    split = str(training.get("evaluation_split", "validation"))
    if split not in {"train", "validation"}:
        raise ValueError("evaluation_split must be 'train' or 'validation'")
    return split


def _model_config(raw_config: dict, tokenizer: HindiTokenizer) -> ModelConfig:
    values = dict(raw_config["model"])
    values["semantic_vocab_size"] = tokenizer.vocabulary_size
    values.update(
        pad_token_id=tokenizer.special_ids.pad,
        bos_token_id=tokenizer.special_ids.bos,
        eos_token_id=tokenizer.special_ids.eos,
    )
    return ModelConfig(**values)


def _device_name(requested: str) -> str:
    return detect_runtime().device if requested == "auto" else requested


def _save_checkpoint(
    path: Path,
    *,
    model: SpeechSemanticModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_validation_loss: float,
    train_history: list[dict],
    validation_history: list[dict],
    generation_history: list[dict],
    raw_config: dict,
    manifest_sha256: str,
    tokenizer_sha256: str,
) -> None:
    payload = {
        "step": step,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_history": train_history,
        "validation_history": validation_history,
        "generation_history": generation_history,
        "raw_config": raw_config,
        "manifest_sha256": manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validate_resume(
    checkpoint: dict,
    *,
    raw_config: dict,
    manifest_sha256: str,
    tokenizer_sha256: str,
) -> None:
    if checkpoint["raw_config"]["model"] != raw_config["model"]:
        raise ValueError("resume checkpoint model configuration differs")
    if checkpoint["manifest_sha256"] != manifest_sha256:
        raise ValueError("resume checkpoint manifest differs")
    if checkpoint["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError("resume checkpoint tokenizer differs")


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    metadata = train(
        args.config,
        args.manifest,
        args.tokenizer,
        args.output,
        resume_path=args.resume,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
