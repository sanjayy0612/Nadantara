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

The initial shortlist is documented in `docs/dataset-audit.md`. Downloaders will be
added only after the selected source's access terms have been reviewed.
