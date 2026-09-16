# Dataset workspace

This directory stores dataset definitions and lightweight manifests, never raw audio or
large generated artifacts.

```text
dataset/
├── manifests/          # versioned JSONL manifests (small samples may be committed)
├── manifest.schema.json
├── raw/                # ignored downloads
├── processed/          # ignored features, units and shards
└── cache/              # ignored transient data
```

Every row must record provenance, license, real/synthetic origin, split, duration and a
content hash. Never create train/test splits after combining datasets: reserve official
evaluation splits and deduplicate before training.

The evidence-backed shortlist is documented in `docs/dataset-audit.md` and represented
in `catalog.json`. Downloaders will be added only after the selected source's access
terms have been reviewed.

## IndicVoices-R development subset

Place selected Hindi Parquet shards under `raw/indicvoices_r/Hindi/`. Raw files remain
ignored by Git. Install the optional reader dependency and stream examples in bounded
batches:

```bash
python -m pip install -e '.[data]'
python - <<'PY'
from itertools import islice
from seamtranslation.datasets import iter_indicvoices

rows = iter_indicvoices("dataset/raw/indicvoices_r/Hindi")
for row in islice(rows, 3):
    print(row.audio_filename, row.duration_seconds, row.normalized_text)
PY
```

The source audio is embedded in each Parquet row as 48 kHz WAV bytes. Resampling to the
model's 16 kHz boundary contract belongs in preprocessing, not in this source reader.

Prepare all locally downloaded shards with a deterministic speaker-disjoint split:

```bash
PYTHONPATH=src python train/prepare_indicvoices.py \
  --source dataset/raw/indicvoices_r/Hindi \
  --output dataset/processed/indicvoices_r_hi_dev \
  --dataset-revision 5f4495c91d500742a58d1be2ab07d77f73c0acf8
```

The ignored output contains 16 kHz PCM WAV files, `manifest.jsonl`, and a provenance and
split summary in `summary.json`. Refusing to overwrite an existing output prevents two
different preprocessing runs from being silently mixed.
