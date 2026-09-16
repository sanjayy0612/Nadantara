"""Prepare a bounded English-Hindi semantic-decoder pretraining subset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from seamtranslation.preprocessing.samanantar import prepare_samanantar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    args = parser.parse_args()
    summary = prepare_samanantar(
        args.source,
        args.output,
        dataset_revision=args.dataset_revision,
        limit=args.limit,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
