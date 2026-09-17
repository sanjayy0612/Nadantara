"""Train the English-speech-to-Hindi-semantic stage."""

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

from seamtranslation.models import ModelConfig, SeamTranslationModel
from seamtranslation.runtime import detect_runtime
from seamtranslation.tokenization import HindiTokenizer
from seamtranslation.training import (
    WordProjectSemanticDataset,
    collate_semantic_batch,
    semantic_loss,
)


def train(config_path: Path, manifest_path: Path, tokenizer_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    raw_config = json.loads(config_path.read_text())
    training = raw_config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = HindiTokenizer(tokenizer_path)
    model_values = dict(raw_config["model"])
    model_values["semantic_vocab_size"] = tokenizer.vocabulary_size
    model_values.update(
        pad_token_id=tokenizer.special_ids.pad,
        bos_token_id=tokenizer.special_ids.bos,
        eos_token_id=tokenizer.special_ids.eos,
    )
    model_config = ModelConfig(**model_values)
    requested_device = training.get("device", "auto")
    device_name = detect_runtime().device if requested_device == "auto" else requested_device
    device = torch.device(device_name)
    dataset = WordProjectSemanticDataset(
        manifest_path,
        split="train",
        max_examples=int(training["max_examples"]),
    )
    model = SeamTranslationModel(model_config).to(device)
    trainable_modules = (model.frontend, model.subsampler, model.encoder, model.semantic_decoder)
    parameters = [parameter for module in trainable_modules for parameter in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=float(training["learning_rate"]))
    max_steps = int(training["max_steps"])
    batch_size = int(training["batch_size"])
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for step in range(max_steps):
        indices = [(step * batch_size + offset) % len(dataset) for offset in range(batch_size)]
        batch = collate_semantic_batch(
            [dataset[index] for index in indices],
            tokenizer,
            max_tokens=model_config.max_semantic_tokens,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = semantic_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(training["max_gradient_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"step={step + 1} loss={losses[-1]:.6f}")

    output.mkdir(parents=True)
    elapsed = time.perf_counter() - started
    metadata = {
        "git_commit": _git_commit(),
        "config": raw_config,
        "model_config": asdict(model_config),
        "manifest_sha256": _sha256_file(manifest_path),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
        "device": device_name,
        "steps": max_steps,
        "examples": len(dataset),
        "losses": losses,
        "elapsed_seconds": elapsed,
        "parameters_total": model.parameter_count(),
        "parameters_trained": sum(parameter.numel() for parameter in parameters),
    }
    torch.save(
        {
            "step": max_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        output / "checkpoint.pt",
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


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
    args = parser.parse_args()
    metadata = train(args.config, args.manifest, args.tokenizer, args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
