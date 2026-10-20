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

## Audio-dependence diagnostic

Compare a trusted local semantic checkpoint on correct audio and a seeded permutation
of validation audio. Every shuffled example receives another example's audio; Hindi
references remain fixed. This evaluates existing weights without training:

```bash
PYTHONPATH=src:. python train/check_audio_dependence.py \
  --checkpoint train/runs/semantic-2k-m2/best.pt \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --tokenizer artifacts/tokenizers/hindi_2k/hindi.model \
  --examples 264 --device cpu \
  --output train/diagnostics/baseline_audio_dependence.json
```

The JSON records metrics, all references and hypotheses, the audio permutation,
artifact hashes, and the fraction of unchanged outputs. A small quality difference
suggests weak useful audio conditioning; it does not prove the encoder is broken.
This is a development diagnostic, not an independent final test. Only load trusted
checkpoints, since the trainer's checkpoint format requires Python deserialization.

`semantic-2k-local-continuation.json` defines a bounded continuation from the
baseline's step-2900 checkpoint to step 3900. Its optional `lr_schedule` ramps up
for 50 continuation steps and then decays to 10% of the peak learning rate.
`start_step` is an explicit schedule origin, so interruption/resume retains the
same learning-rate progression. Write this experiment into a new run directory.

## Frozen pretrained speech experiment

`compare_pretrained_encoder.py` is a research-only comparison of frozen Whisper
Tiny features with frozen randomly initialized Whisper features. It does not
modify the deployed `SpeechSemanticModel`. Both conditions initialize the Hindi
decoder from the same step-2900 baseline, initialize the adapter identically,
use all 1,736 training examples, and train for the same number of steps. Only
the speech encoder's pretrained versus random weights differ. The random
condition is a control for encoder architecture and capacity, not a competitive
fully trained speech encoder.

Download `config.json`, `preprocessor_config.json`, and `model.safetensors` from
`openai/whisper-tiny` into an ignored local artifact directory, pinning the Hub
revision in a `revision.json` file with `model_id` and `revision` fields.
The script needs Transformers and the existing training dependencies.

```bash
PYTHONPATH=src:. python train/compare_pretrained_encoder.py \
  --encoder artifacts/pretrained/whisper-tiny \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --tokenizer artifacts/tokenizers/hindi_2k/hindi.model \
  --baseline train/runs/semantic-2k-m2/best.pt \
  --output train/runs/frozen-whisper-comparison --steps 1000 --device mps
```

The encoder is frozen and cached once per condition. Only a 384-to-128 adapter
and the existing Hindi decoder are trained. Audio longer than 30 seconds is
rejected instead of silently truncated. The final heads are evaluated on all
264 development examples with correct and shuffled audio. Caches, head weights,
and JSON outputs stay in the ignored run directory. A head checkpoint requires
its pinned encoder and Hindi tokenizer for waveform inference; it cannot be
loaded by the baseline model's checkpoint loader.

This tests whether pretrained representations help under a bounded training
budget. It increases total parameters and does not establish deployability,
export compatibility, or performance after fine-tuning the encoder. It is a
single-seed development experiment, not an independent test-set result.

Resume an interrupted run into the same directory:

```bash
PYTHONPATH=src:. .venv/bin/python train/train_semantic.py \
  --config train/configs/semantic-2k-m2.json \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --tokenizer artifacts/tokenizers/hindi_2k/hindi.model \
  --output train/runs/semantic-2k-m2 \
  --resume train/runs/semantic-2k-m2/checkpoint.pt
```
