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

## Corrected 2K-token training sequence

The initial 1,000-step run used an 8K Hindi vocabulary. Its validation loss decreased,
but greedy decoding collapsed into repeated common tokens. Do not continue that
checkpoint: changing vocabulary size changes the output projection, so the corrected
2K-token experiment must start from a new model.

Install the complete training dependency set after pulling these changes:

```bash
.venv/bin/python -m pip install -e '.[training]'
```

The ignored tokenizer artifact can be rebuilt deterministically from the two local
manifests. Its expected model SHA-256 is recorded in
`model_registry/tokenizers/hindi_2k.json`:

```bash
PYTHONPATH=src:. .venv/bin/python train/train_tokenizer.py \
  --manifest dataset/processed/samanantar_hi_100k/manifest.jsonl \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --output artifacts/tokenizers/hindi_2k \
  --vocabulary-size 2000 \
  --text-field target_text
```

First run the 16-example memorization gate. This intentionally evaluates the same 16
training examples; it tests model wiring and optimization, not generalization:

```bash
PYTHONPATH=src:. .venv/bin/python train/train_semantic.py \
  --config train/configs/semantic-overfit-16-m2.json \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --tokenizer artifacts/tokenizers/hindi_2k/hindi.model \
  --output train/runs/semantic-overfit-16-m2
```

Inspect the last `generation_history` entry in the resulting `metadata.json`. Proceed
only if the generated samples reproduce the references, EOS rate is 1.0, exact-match
rate is close to 1.0, and adjacent repetition is not visibly pathological. A failure to
memorize 16 examples is a training-pipeline problem; more full-corpus steps will not fix
it.

After that gate passes, start the corrected full run. It uses all 1,736 training
examples, holds out 264 validation examples, evaluates actual generations every 250
steps, and checkpoints every 500 steps to avoid the previous checkpoint explosion:

```bash
PYTHONPATH=src:. .venv/bin/python train/train_semantic.py \
  --config train/configs/semantic-2k-m2.json \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --tokenizer artifacts/tokenizers/hindi_2k/hindi.model \
  --output train/runs/semantic-2k-m2
```

The historical `semantic-full-m2*.json` files preserve the original 8K-token
experiments; they are not the recommended next run.

Resume an interrupted run into the same directory:

```bash
PYTHONPATH=src:. .venv/bin/python train/train_semantic.py \
  --config train/configs/semantic-2k-m2.json \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --tokenizer artifacts/tokenizers/hindi_2k/hindi.model \
  --output train/runs/semantic-2k-m2 \
  --resume train/runs/semantic-2k-m2/checkpoint.pt
```
