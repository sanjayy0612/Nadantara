# Training workspace

Training entry points and versioned experiment configurations live here. Large outputs
are written to `train/runs/`, which is ignored.

The training code must:

- read a validated manifest rather than scan arbitrary directories;
- record the Git commit, config, model-registry IDs and manifest hash;
- set random seeds and report nondeterministic operations;
- separate teacher-target extraction from student training;
- save resumable checkpoints outside Git;
- never label synthetic translations or speech as human data.

`train/configs/tiny-smoke.json` is deliberately too small for useful translation. It
exists to exercise the training graph on CPU/MPS without exhausting the 8 GB Mac.

The first real stage trains English speech to Hindi semantic tokens independently of
the unit decoder and vocoder:

```bash
PYTHONPATH=src:. .venv/bin/python train/train_semantic.py \
  --config train/configs/semantic-smoke.json \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --tokenizer artifacts/tokenizers/hindi_8k/hindi.model \
  --output train/runs/semantic-smoke
```

This configuration is a short graph and checkpoint verification run, not a quality
training run. Increase steps and model dimensions only after it finishes with finite,
decreasing loss and a reloadable checkpoint.

For the first bounded M2 experiment, use `train/configs/semantic-local-m2.json`. It
trains a semantic-only model for 200 steps, evaluates every 25 steps, saves every 50
steps, and keeps the best validation checkpoint:

```bash
PYTHONPATH=src:. .venv/bin/python train/train_semantic.py \
  --config train/configs/semantic-local-m2.json \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --tokenizer artifacts/tokenizers/hindi_8k/hindi.model \
  --output train/runs/semantic-local-m2
```

Resume an interrupted run into the same directory:

```bash
PYTHONPATH=src:. .venv/bin/python train/train_semantic.py \
  --config train/configs/semantic-local-m2.json \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --tokenizer artifacts/tokenizers/hindi_8k/hindi.model \
  --output train/runs/semantic-local-m2 \
  --resume train/runs/semantic-local-m2/checkpoint.pt
```
