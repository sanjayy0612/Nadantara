"""Memory-bounded reader for local Samanantar English-Hindi Parquet shards."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ParallelTextExample:
    source_text: str
    target_text: str
    source_index: int
    source_shard: str
    source_row: int


def iter_samanantar(
    paths: Path | str | Sequence[Path | str], *, batch_size: int = 1_024
) -> Iterator[ParallelTextExample]:
    """Yield English-Hindi pairs without loading a complete shard into memory."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Samanantar reading requires: python -m pip install -e '.[data]'") from exc

    shards = _resolve_shards(paths)
    if not shards:
        raise FileNotFoundError("no Parquet shards found")
    for shard in shards:
        parquet_file = parquet.ParquetFile(shard)
        missing = {"idx", "src", "tgt"}.difference(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{shard} is missing required columns: {', '.join(sorted(missing))}")
        source_row = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["idx", "src", "tgt"]):
            for row in batch.to_pylist():
                yield ParallelTextExample(
                    source_text=row["src"],
                    target_text=row["tgt"],
                    source_index=int(row["idx"]),
                    source_shard=shard.name,
                    source_row=source_row,
                )
                source_row += 1


def _resolve_shards(paths: Path | str | Sequence[Path | str]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidate = Path(paths)
        candidates = sorted(candidate.glob("*.parquet")) if candidate.is_dir() else [candidate]
    else:
        candidates = sorted(Path(path) for path in paths)
    for candidate in candidates:
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.suffix != ".parquet":
            raise ValueError(f"expected a .parquet shard, got {candidate}")
    return candidates
