"""Memory-bounded reader for WordProject English-to-Indic speech translation shards."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WordProjectExample:
    audio_bytes: bytes
    source_text: str
    source_asr_text: str
    target_text: str
    recording_id: str
    duration_seconds: float
    alignment_score: float
    translation_score: float
    source_shard: str
    source_row: int


def iter_wordproject(path: Path | str, *, batch_size: int = 128) -> Iterator[WordProjectExample]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("WordProject reading requires: python -m pip install -e '.[data]'") from exc

    source = Path(path)
    shards = sorted(source.glob("*.parquet")) if source.is_dir() else [source]
    if not shards:
        raise FileNotFoundError("no Parquet shards found")
    columns = [
        "chunked_audio_filepath", "text", "pred_text", "hi_text", "audio_filepath",
        "duration", "alignment_score", "hi_mining_score",
    ]
    for shard in shards:
        if not shard.is_file():
            raise FileNotFoundError(shard)
        parquet_file = parquet.ParquetFile(shard)
        missing = set(columns).difference(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{shard} is missing required columns: {', '.join(sorted(missing))}")
        source_row = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield WordProjectExample(
                    audio_bytes=row["chunked_audio_filepath"]["bytes"],
                    source_text=row["text"],
                    source_asr_text=row["pred_text"],
                    target_text=row["hi_text"],
                    recording_id=row["audio_filepath"],
                    duration_seconds=float(row["duration"]),
                    alignment_score=float(row["alignment_score"]),
                    translation_score=float(row["hi_mining_score"]),
                    source_shard=shard.name,
                    source_row=source_row,
                )
                source_row += 1
