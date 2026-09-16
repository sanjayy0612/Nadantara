"""Filter and version a bounded English-Hindi Samanantar development subset."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

from seamtranslation.datasets import iter_samanantar


DATASET_ID = "ai4bharat/samanantar"
DATASET_URL = "https://huggingface.co/datasets/ai4bharat/samanantar"
DATASET_LICENSE = "CC0-1.0 packaging; underlying-source rights vary"
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ParallelTextSummary:
    dataset_id: str
    dataset_revision: str
    source_shard_sha256: dict[str, str]
    scanned: int
    accepted: int
    split_examples: dict[str, int]
    rejected: dict[str, int]


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def rejection_reason(source: str, target: str) -> str | None:
    """Return the first deterministic quality-filter failure, or ``None``."""

    if not source or not target:
        return "empty"
    if not 5 <= len(source) <= 500 or not 3 <= len(target) <= 500:
        return "length"
    ratio = len(target) / len(source)
    if not 0.3 <= ratio <= 3.0:
        return "length_ratio"
    source_letters = [char for char in source if char.isalpha()]
    target_letters = [char for char in target if char.isalpha()]
    if not source_letters or not target_letters:
        return "no_letters"
    ascii_latin = sum("a" <= char.lower() <= "z" for char in source_letters)
    devanagari = sum("\u0900" <= char <= "\u097f" for char in target_letters)
    if ascii_latin / len(source_letters) < 0.7:
        return "source_script"
    if devanagari / len(target_letters) < 0.7:
        return "target_script"
    return None


def prepare_samanantar(
    source: Path,
    output: Path,
    *,
    dataset_revision: str,
    limit: int = 100_000,
    validation_fraction: float = 0.01,
) -> ParallelTextSummary:
    """Write filtered, deduplicated JSONL pairs with a deterministic monitor split."""

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not dataset_revision:
        raise ValueError("dataset_revision is required")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    output.mkdir(parents=True)
    shard_paths = sorted(source.glob("*.parquet"))
    shard_hashes = {path.name: _sha256_file(path) for path in shard_paths}
    seen_source: set[str] = set()
    seen_target: set[str] = set()
    rejected: Counter[str] = Counter()
    split_examples: Counter[str] = Counter()
    scanned = 0
    accepted = 0

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for example in iter_samanantar(source):
            scanned += 1
            source_text = normalize_text(example.source_text)
            target_text = normalize_text(example.target_text)
            reason = rejection_reason(source_text, target_text)
            if reason:
                rejected[reason] += 1
                continue
            source_key = source_text.casefold()
            target_key = target_text.casefold()
            if source_key in seen_source or target_key in seen_target:
                rejected["duplicate"] += 1
                continue

            pair_hash = sha256(f"{source_text}\n{target_text}".encode()).hexdigest()
            bucket = int(pair_hash[:16], 16) / 2**64
            split = "validation" if bucket < validation_fraction else "train"
            row = {
                "id": f"samanantar-hi-{pair_hash[:20]}",
                "dataset_id": DATASET_ID,
                "dataset_revision": dataset_revision,
                "source_url": DATASET_URL,
                "source_license": DATASET_LICENSE,
                "source_subset": "hi",
                "source_shard": example.source_shard,
                "source_row": example.source_row,
                "source_index": example.source_index,
                "source_language": "en",
                "target_language": "hi",
                "source_text": source_text,
                "target_text": target_text,
                "translation_origin": "mixed-existing-and-mined",
                "split": split,
                "content_sha256": pair_hash,
            }
            manifest.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            seen_source.add(source_key)
            seen_target.add(target_key)
            split_examples[split] += 1
            accepted += 1
            if accepted >= limit:
                break

    summary = ParallelTextSummary(
        dataset_id=DATASET_ID,
        dataset_revision=dataset_revision,
        source_shard_sha256=shard_hashes,
        scanned=scanned,
        accepted=accepted,
        split_examples=dict(sorted(split_examples.items())),
        rejected=dict(sorted(rejected.items())),
    )
    (output / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
