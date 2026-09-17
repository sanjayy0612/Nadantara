"""Prepare English-speech-to-Hindi-text WordProject data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from seamtranslation.preprocessing.wordproject import prepare_wordproject


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--limit", type=int, default=2_000)
    parser.add_argument("--minimum-alignment-score", type=float, default=0.9)
    parser.add_argument("--minimum-translation-score", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    summary = prepare_wordproject(
        args.source,
        args.output,
        dataset_revision=args.dataset_revision,
        limit=args.limit,
        minimum_alignment_score=args.minimum_alignment_score,
        minimum_translation_score=args.minimum_translation_score,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
