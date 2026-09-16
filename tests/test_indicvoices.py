from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seamtranslation.datasets import iter_indicvoices


def _write_shard(path: Path, *, language: str = "hi") -> None:
    table = pa.table(
        {
            "text": ["नमस्ते"],
            "normalized": ["नमस्ते"],
            "audio": pa.array(
                [{"bytes": b"RIFF-test", "path": "sample.wav"}],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "speaker_id": ["speaker-1"],
            "duration": [1.25],
            "samples": [60_000],
            "lang": [language],
        }
    )
    pq.write_table(table, path)


def test_iter_indicvoices_reads_embedded_audio(tmp_path: Path) -> None:
    _write_shard(tmp_path / "train-00000.parquet")

    examples = list(iter_indicvoices(tmp_path, batch_size=1))

    assert len(examples) == 1
    assert examples[0].normalized_text == "नमस्ते"
    assert examples[0].audio_bytes == b"RIFF-test"
    assert examples[0].audio_filename == "sample.wav"
    assert examples[0].duration_seconds == 1.25
    assert examples[0].source_shard == "train-00000.parquet"
    assert examples[0].source_row == 0


def test_iter_indicvoices_rejects_wrong_language(tmp_path: Path) -> None:
    _write_shard(tmp_path / "train-00000.parquet", language="en")

    with pytest.raises(ValueError, match="expected language 'hi', got 'en'"):
        next(iter_indicvoices(tmp_path))


def test_iter_indicvoices_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no Parquet shards found"):
        next(iter_indicvoices(tmp_path))
