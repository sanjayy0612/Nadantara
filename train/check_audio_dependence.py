"""Compare a semantic checkpoint on matched and deterministically shuffled audio."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from seamtranslation.evaluation import semantic_generation_metrics
from seamtranslation.models import SpeechSemanticModel
from seamtranslation.tokenization import HindiTokenizer
from seamtranslation.training import WordProjectSemanticDataset
from train.train_semantic import _generated_content_tokens, _model_config


def audio_permutation(size: int, seed: int) -> list[int]:
    """Shuffle into one cycle so every reference receives another example's audio."""
    if size < 2:
        raise ValueError("audio comparison requires at least two examples")
    order = list(range(size))
    random.Random(seed).shuffle(order)
    permutation = [0] * size
    for position, index in enumerate(order):
        permutation[index] = order[(position + 1) % size]
    return permutation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.examples < 2:
        parser.error("--examples must be at least 2")
    torch.set_num_threads(4)
    tokenizer = HindiTokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer_hash = sha256(args.tokenizer.read_bytes()).hexdigest()
    if tokenizer_hash != checkpoint["tokenizer_sha256"]:
        raise ValueError("tokenizer hash differs from checkpoint")
    model = SpeechSemanticModel(_model_config(checkpoint["raw_config"], tokenizer))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device).eval()
    dataset = WordProjectSemanticDataset(
        args.manifest, split="validation", max_examples=args.examples
    )
    examples = [dataset[index] for index in range(len(dataset))]
    permutation = audio_permutation(len(examples), args.seed)
    references = [example.target_text for example in examples]
    conditions = {}
    for name, indices in (("correct_audio", list(range(len(examples)))),
                          ("shuffled_audio", permutation)):
        hypotheses, token_rows, eos_flags = [], [], []
        with torch.inference_mode():
            for position, audio_index in enumerate(indices):
                waveform = examples[audio_index].waveform.to(args.device)
                lengths = torch.tensor([waveform.numel()], device=args.device)
                row = model.generate(waveform.unsqueeze(0), lengths)[0].cpu().tolist()
                tokens, ended = _generated_content_tokens(row, tokenizer)
                hypotheses.append(tokenizer.decode(tokens))
                token_rows.append(tokens)
                eos_flags.append(ended)
                print(f"{name}: {position + 1}/{len(indices)}", flush=True)
        conditions[name] = {
            "metrics": semantic_generation_metrics(
                hypotheses, references, token_rows, eos_flags
            ),
            "hypotheses": hypotheses,
        }
    correct = conditions["correct_audio"]
    shuffled = conditions["shuffled_audio"]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint["step"],
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "tokenizer_sha256": tokenizer_hash,
        "device": args.device,
        "torch_version": torch.__version__,
        "seed": args.seed,
        "shuffled_audio_indices": permutation,
        "references": references,
        "conditions": conditions,
        "correct_minus_shuffled": {
            metric: correct["metrics"][metric] - shuffled["metrics"][metric]
            for metric in ("chrf2", "bleu")
        },
        "unchanged_output_rate": sum(
            a == b for a, b in zip(correct["hypotheses"], shuffled["hypotheses"], strict=True)
        ) / len(examples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("correct_minus_shuffled", "unchanged_output_rate")}, indent=2))


if __name__ == "__main__":
    main()
