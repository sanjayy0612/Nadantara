"""Prepare high-confidence English-speech-to-Hindi-text WordProject data."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from seamtranslation.datasets import iter_wordproject
from seamtranslation.preprocessing.audio import decode_compressed_audio, encode_wav, resample
from seamtranslation.preprocessing.samanantar import normalize_text, rejection_reason


DATASET_ID = "ai4bharat/WordProject"
DATASET_URL = "https://huggingface.co/datasets/ai4bharat/WordProject"
DATASET_LICENSE = "CC-BY-4.0"


@dataclass(frozen=True, slots=True)
class WordProjectSummary:
    dataset_id: str
    dataset_revision: str
    source_shard_sha256: dict[str, str]
    scanned: int
    accepted: int
    duration_seconds: float
    split_examples: dict[str, int]
    split_recordings: dict[str, int]
    rejected: dict[str, int]
    sample_rate: int


def prepare_wordproject(
    source: Path,
    output: Path,
    *,
    dataset_revision: str,
    limit: int = 2_000,
    minimum_alignment_score: float = 0.9,
    minimum_translation_score: float = 0.8,
    validation_fraction: float = 0.1,
) -> WordProjectSummary:
    """Filter, decode, split by source recording, and write a local training subset."""

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not dataset_revision:
        raise ValueError("dataset_revision is required")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True)
    shard_paths = sorted(source.glob("*.parquet"))
    shard_hashes = {path.name: _sha256_file(path) for path in shard_paths}
    rejected: Counter[str] = Counter()
    split_examples: Counter[str] = Counter()
    split_recordings: dict[str, set[str]] = {"train": set(), "validation": set()}
    seen_audio: set[str] = set()
    scanned = 0
    accepted = 0
    duration_seconds = 0.0

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for example in iter_wordproject(source):
            scanned += 1
            if not 1.0 <= example.duration_seconds <= 20.0:
                rejected["duration"] += 1
                continue
            if example.alignment_score < minimum_alignment_score:
                rejected["alignment_score"] += 1
                continue
            if example.translation_score < minimum_translation_score:
                rejected["translation_score"] += 1
                continue

            source_text = normalize_text(example.source_text)
            target_text = normalize_text(example.target_text)
            if reason := rejection_reason(source_text, target_text):
                rejected[reason] += 1
                continue

            original_audio_sha256 = sha256(example.audio_bytes).hexdigest()
            if original_audio_sha256 in seen_audio:
                rejected["duplicate_audio"] += 1
                continue
            converted = resample(decode_compressed_audio(example.audio_bytes), target_rate=16_000)
            wav_bytes = encode_wav(converted)
            content_sha256 = sha256(wav_bytes).hexdigest()
            relative_audio_path = Path("audio") / f"{content_sha256}.wav"
            (output / relative_audio_path).write_bytes(wav_bytes)

            recording_hash = sha256(example.recording_id.encode()).hexdigest()
            bucket = int(recording_hash[:16], 16) / 2**64
            split = "validation" if bucket < validation_fraction else "train"
            identity = sha256(
                f"{example.source_shard}:{example.source_row}:{content_sha256}".encode()
            ).hexdigest()
            observed_duration = len(converted.waveform) / converted.sample_rate
            row = {
                "id": f"wordproject-en-hi-{identity[:20]}",
                "dataset_id": DATASET_ID,
                "dataset_revision": dataset_revision,
                "source_url": DATASET_URL,
                "source_license": DATASET_LICENSE,
                "source_subset": "En-Indic/hi",
                "source_shard": example.source_shard,
                "source_row": example.source_row,
                "recording_hash": recording_hash,
                "audio_path": relative_audio_path.as_posix(),
                "audio_origin": "real",
                "source_language": "en",
                "target_language": "hi",
                "source_text": source_text,
                "source_asr_text": normalize_text(example.source_asr_text),
                "target_text": target_text,
                "translation_origin": "mined",
                "duration_seconds": observed_duration,
                "sample_rate": converted.sample_rate,
                "alignment_score": example.alignment_score,
                "translation_score": example.translation_score,
                "split": split,
                "content_sha256": content_sha256,
                "original_audio_sha256": original_audio_sha256,
            }
            manifest.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            seen_audio.add(original_audio_sha256)
            split_examples[split] += 1
            split_recordings[split].add(recording_hash)
            duration_seconds += observed_duration
            accepted += 1
            if accepted >= limit:
                break

    if split_recordings["train"].intersection(split_recordings["validation"]):
        raise RuntimeError("source-recording leakage detected")
    summary = WordProjectSummary(
        dataset_id=DATASET_ID,
        dataset_revision=dataset_revision,
        source_shard_sha256=shard_hashes,
        scanned=scanned,
        accepted=accepted,
        duration_seconds=duration_seconds,
        split_examples=dict(sorted(split_examples.items())),
        split_recordings={key: len(value) for key, value in split_recordings.items()},
        rejected=dict(sorted(rejected.items())),
        sample_rate=16_000,
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
