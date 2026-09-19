"""Create the small Nadantara source bundle required by the Colab scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile


FILES = (
    "pyproject.toml",
    "src",
    "train/train_semantic.py",
    "train/configs/semantic-2k-m2.json",
    "dataset/processed/wordproject_en_hi_2k/manifest.jsonl",
    "artifacts/tokenizers/hindi_2k/hindi.model",
    "experiments/distillation/held_out_benchmark_ids.txt",
    "artifacts/teacher_benchmarks/wordproject_validation_100_seed_20260918/benchmark_manifest.jsonl",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def include(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(member.name).parts
        if "__pycache__" in parts or parts[-1] == ".DS_Store":
            return None
        return member

    with tarfile.open(args.output, "w:gz") as archive:
        for relative in FILES:
            path = root / relative
            if not path.exists():
                raise FileNotFoundError(path)
            archive.add(path, arcname=Path("Nadantara") / relative, filter=include)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
