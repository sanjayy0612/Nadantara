from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile

from seamtranslation.preprocessing.wordproject import prepare_wordproject


def _flac(value: float) -> bytes:
    output = BytesIO()
    soundfile.write(output, np.full(16_000, value, dtype=np.float32), 16_000, format="FLAC")
    return output.getvalue()


def _write_shard(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "chunked_audio_filepath": pa.array(
                    [
                        {"bytes": _flac(0.1), "path": "one.wav"},
                        {"bytes": _flac(0.2), "path": "two.wav"},
                        {"bytes": _flac(0.3), "path": "three.wav"},
                    ],
                    type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
                ),
                "text": ["A valid English sentence."] * 3,
                "pred_text": ["A valid English sentence."] * 3,
                "hi_text": ["यह एक सही हिन्दी वाक्य है।"] * 3,
                "audio_filepath": ["recording-a", "recording-a", "recording-b"],
                "duration": [1.0, 1.0, 1.0],
                "alignment_score": [0.95, 0.95, 0.2],
                "hi_mining_score": [0.9, 0.9, 0.9],
            }
        ),
        path,
    )


def test_prepare_wordproject_filters_and_groups_recordings(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_shard(source / "train.parquet")
    output = tmp_path / "output"

    summary = prepare_wordproject(source, output, dataset_revision="revision", limit=10)

    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert summary.scanned == 3
    assert summary.accepted == 2
    assert summary.rejected == {"alignment_score": 1}
    assert len({row["split"] for row in rows}) == 1
    assert all(row["sample_rate"] == 16_000 for row in rows)
    assert all((output / row["audio_path"]).is_file() for row in rows)
