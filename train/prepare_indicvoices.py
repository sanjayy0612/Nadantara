"""Prepare local IndicVoices-R Hindi shards for unit/vocoder training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from seamtranslation.preprocessing.indicvoices import prepare_indicvoices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="seamtranslation-v1")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    summary = prepare_indicvoices(
        args.source,
        args.output,
        dataset_revision=args.dataset_revision,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
