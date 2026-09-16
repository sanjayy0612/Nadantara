import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from seamtranslation.datasets import iter_samanantar
from seamtranslation.preprocessing.samanantar import prepare_samanantar, rejection_reason


def _write_shard(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "idx": [1, 2, 3],
                "src": ["A valid English sentence.", "A valid English sentence.", "नमस्ते"],
                "tgt": ["यह एक सही हिन्दी वाक्य है।", "एक दूसरा अनुवाद।", "Hello"],
            }
        ),
        path,
    )


def test_iter_samanantar_reads_rows(tmp_path: Path) -> None:
    _write_shard(tmp_path / "train.parquet")
    rows = list(iter_samanantar(tmp_path, batch_size=2))
    assert len(rows) == 3
    assert rows[0].source_index == 1
    assert rows[2].source_row == 2


def test_rejection_reason_checks_scripts() -> None:
    assert rejection_reason("A valid English sentence.", "यह सही हिन्दी है।") is None
    assert rejection_reason("यह अंग्रेज़ी नहीं है", "यह हिन्दी है।") == "source_script"
    assert rejection_reason("This is English.", "This is not Hindi.") == "target_script"


def test_prepare_samanantar_filters_and_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_shard(source / "train.parquet")
    output = tmp_path / "output"

    summary = prepare_samanantar(source, output, dataset_revision="revision", limit=10)

    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert summary.scanned == 3
    assert summary.accepted == 1
    assert summary.rejected == {"duplicate": 1, "source_script": 1}
    assert rows[0]["source_language"] == "en"
    assert rows[0]["target_language"] == "hi"
