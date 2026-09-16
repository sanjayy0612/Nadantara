"""Build a reproducible, speaker-disjoint Hindi speech training subset."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from seamtranslation.datasets import iter_indicvoices
from seamtranslation.preprocessing.audio import decode_wav, encode_wav, resample


DATASET_ID = "ai4bharat/indicvoices_r"
DATASET_URL = "https://huggingface.co/datasets/ai4bharat/indicvoices_r"
DATASET_LICENSE = "CC-BY-4.0"


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    dataset_id: str
    dataset_revision: str
    source_shards: tuple[str, ...]
    source_shard_sha256: dict[str, str]
    examples: int
    speakers: int
    duration_seconds: float
    split_examples: dict[str, int]
    split_speakers: dict[str, int]
    sample_rate: int


def assign_speaker_split(
    speaker_id: str,
    *,
    validation_fraction: float = 0.1,
    seed: str = "seamtranslation-v1",
) -> str:
    """Assign all utterances from a speaker to one stable split."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    digest = sha256(f"{seed}:{speaker_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if bucket < validation_fraction else "train"


def prepare_indicvoices(
    source: Path,
    output: Path,
    *,
    dataset_revision: str,
    validation_fraction: float = 0.1,
    seed: str = "seamtranslation-v1",
    limit: int | None = None,
) -> PreparationSummary:
    """Resample local shards and write a JSONL manifest plus provenance summary."""

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not dataset_revision:
        raise ValueError("dataset_revision is required")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True)
    manifest_path = output / "manifest.jsonl"
    split_examples: Counter[str] = Counter()
    split_speakers: dict[str, set[str]] = {"train": set(), "validation": set()}
    speakers: set[str] = set()
    duration_seconds = 0.0
    source_shards: set[str] = set()
    shard_paths = sorted(source.glob("*.parquet"))
    source_shard_sha256 = {path.name: _sha256_file(path) for path in shard_paths}

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, example in enumerate(iter_indicvoices(source)):
            if limit is not None and index >= limit:
                break

            split = assign_speaker_split(
                example.speaker_id,
                validation_fraction=validation_fraction,
                seed=seed,
            )
            converted = resample(decode_wav(example.audio_bytes), target_rate=16_000)
            wav_bytes = encode_wav(converted)
            content_sha256 = sha256(wav_bytes).hexdigest()
            original_sha256 = sha256(example.audio_bytes).hexdigest()
            identity = sha256(
                f"{example.source_shard}:{example.source_row}:{content_sha256}".encode()
            ).hexdigest()
            relative_audio_path = Path("audio") / f"{content_sha256}.wav"
            (output / relative_audio_path).write_bytes(wav_bytes)

            row = {
                "id": f"indicvoices-r-hi-{identity[:20]}",
                "dataset_id": DATASET_ID,
                "dataset_revision": dataset_revision,
                "source_url": DATASET_URL,
                "source_license": DATASET_LICENSE,
                "source_subset": "Hindi",
                "source_shard": example.source_shard,
                "source_row": example.source_row,
                "audio_path": relative_audio_path.as_posix(),
                "audio_origin": "enhanced-real",
                "language": "hi",
                "text": example.text,
                "normalized_text": example.normalized_text,
                "speaker_id": example.speaker_id,
                "duration_seconds": len(converted.waveform) / converted.sample_rate,
                "sample_rate": converted.sample_rate,
                "split": split,
                "content_sha256": content_sha256,
                "original_audio_sha256": original_sha256,
            }
            manifest.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

            split_examples[split] += 1
            split_speakers[split].add(example.speaker_id)
            speakers.add(example.speaker_id)
            duration_seconds += row["duration_seconds"]
            source_shards.add(example.source_shard)

    if split_speakers["train"].intersection(split_speakers["validation"]):
        raise RuntimeError("speaker leakage detected between train and validation")

    summary = PreparationSummary(
        dataset_id=DATASET_ID,
        dataset_revision=dataset_revision,
        source_shards=tuple(sorted(source_shards)),
        source_shard_sha256=source_shard_sha256,
        examples=sum(split_examples.values()),
        speakers=len(speakers),
        duration_seconds=duration_seconds,
        split_examples=dict(sorted(split_examples.items())),
        split_speakers={key: len(value) for key, value in split_speakers.items()},
        sample_rate=16_000,
    )
    (output / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
