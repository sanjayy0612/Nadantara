"""Memory-bounded reader for local IndicVoices-R Parquet shards."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IndicVoicesExample:
    """One Hindi speech example with embedded WAV bytes and provenance metadata."""

    text: str
    normalized_text: str
    audio_bytes: bytes
    audio_filename: str
    speaker_id: str
    duration_seconds: float
    sample_count: int
    language: str
    source_shard: str
    source_row: int


_COLUMNS = (
    "text",
    "normalized",
    "audio",
    "speaker_id",
    "duration",
    "samples",
    "lang",
)


def iter_indicvoices(
    paths: Path | str | Sequence[Path | str],
    *,
    batch_size: int = 32,
    expected_language: str = "hi",
) -> Iterator[IndicVoicesExample]:
    """Yield examples from one or more shards without loading a whole shard into memory.

    ``paths`` may be a Parquet file, a directory containing Parquet files, or an explicit
    sequence. Directories and sequences are read in deterministic path order.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - depends on the caller's environment
        raise RuntimeError(
            "IndicVoices-R reading requires the data extra: "
            "python -m pip install -e '.[data]'"
        ) from exc

    shard_paths = _resolve_shards(paths)
    if not shard_paths:
        raise FileNotFoundError("no Parquet shards found")

    for shard_path in shard_paths:
        parquet_file = parquet.ParquetFile(shard_path)
        missing = set(_COLUMNS).difference(parquet_file.schema_arrow.names)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{shard_path} is missing required columns: {names}")

        source_row = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=_COLUMNS):
            for row in batch.to_pylist():
                language = row["lang"]
                if language != expected_language:
                    raise ValueError(
                        f"expected language {expected_language!r}, got {language!r} "
                        f"in {shard_path}"
                    )

                audio = row["audio"]
                yield IndicVoicesExample(
                    text=row["text"],
                    normalized_text=row["normalized"],
                    audio_bytes=audio["bytes"],
                    audio_filename=audio["path"],
                    speaker_id=row["speaker_id"],
                    duration_seconds=float(row["duration"]),
                    sample_count=int(row["samples"]),
                    language=language,
                    source_shard=shard_path.name,
                    source_row=source_row,
                )
                source_row += 1


def _resolve_shards(paths: Path | str | Sequence[Path | str]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidate = Path(paths)
        if candidate.is_dir():
            return sorted(candidate.glob("*.parquet"))
        candidates = [candidate]
    else:
        candidates = sorted(Path(path) for path in paths)

    for candidate in candidates:
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.suffix != ".parquet":
            raise ValueError(f"expected a .parquet shard, got {candidate}")
    return candidates
