from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import wave

import pyarrow as pa
import pyarrow.parquet as pq

from seamtranslation.preprocessing.indicvoices import (
    assign_speaker_split,
    prepare_indicvoices,
)


def _wav_bytes() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x00\x00" * 4_800)
    return output.getvalue()


def _write_shard(path: Path) -> None:
    audio = _wav_bytes()
    table = pa.table(
        {
            "text": ["एक", "दो"],
            "normalized": ["एक", "दो"],
            "audio": pa.array(
                [
                    {"bytes": audio, "path": "one.wav"},
                    {"bytes": audio, "path": "two.wav"},
                ],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "speaker_id": ["speaker-1", "speaker-2"],
            "duration": [0.1, 0.1],
            "samples": [4_800, 4_800],
            "lang": ["hi", "hi"],
        }
    )
    pq.write_table(table, path)


def test_speaker_split_is_stable_and_seeded() -> None:
    first = assign_speaker_split("speaker-1", validation_fraction=0.5, seed="a")
    assert first == assign_speaker_split("speaker-1", validation_fraction=0.5, seed="a")
    assert first in {"train", "validation"}


def test_prepare_indicvoices_writes_resampled_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_shard(source / "train-00000.parquet")
    output = tmp_path / "output"

    summary = prepare_indicvoices(
        source,
        output,
        dataset_revision="revision-1",
        validation_fraction=0.5,
    )

    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert summary.examples == 2
    assert summary.sample_rate == 16_000
    assert {row["speaker_id"] for row in rows} == {"speaker-1", "speaker-2"}
    assert all(row["sample_rate"] == 16_000 for row in rows)
    for row in rows:
        audio_path = output / row["audio_path"]
        assert audio_path.is_file()
        assert sha256(audio_path.read_bytes()).hexdigest() == row["content_sha256"]
