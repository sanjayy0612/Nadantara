# Teacher experiments

Teacher generation is isolated from the deployable Nadantara student. Large
checkpoints, caches, generated labels, and audio artifacts stay outside Git.

The two notebooks are independently runnable and intentionally do not import
repository Python modules. Each has its own dependency cell, runtime probe,
resumable JSONL writer, and model lifecycle:

- `whisper_indictrans2_teacher.ipynb`: Whisper English transcription, unload,
  then IndicTrans2 `eng_Latn → hin_Deva` translation.
- `seamless_m4t_teacher.ipynb`: SeamlessM4T v2 English speech → Hindi text.

Both consume the same portable `benchmark_manifest.jsonl` and emit the same
canonical teacher-output fields. Human Hindi references are evaluation-only.

## Local bundle preparation

Start with a smoke bundle of five validation examples. The source manifest is not
modified, and the helper verifies audio paths and records SHA-256 hashes:

```bash
PYTHONPATH=. python3 experiments/teachers/prepare_benchmark_bundle.py \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --output /tmp/nadantara-benchmark-smoke \
  --archive /tmp/nadantara-benchmark-smoke.tar.gz \
  --split validation --limit 5 --seed 20260918
```

Use `--limit 150` only after the Phase 0 gate succeeds and you explicitly approve
the teacher smoke tests. Keep the archive outside Git.

## Interactive Colab CLI flow

The installed CLI is version 0.6.0. The browser workflow below avoids its current
`colab exec`/`jupyter-kernel-client` incompatibility and lets you run notebook cells
interactively:

```bash
colab new -s nadantara-whisper --gpu T4
colab upload -s nadantara-whisper /tmp/nadantara-benchmark-smoke.tar.gz /content/nadantara-benchmark-smoke.tar.gz
colab upload -s nadantara-whisper experiments/teachers/whisper_indictrans2_teacher.ipynb /content/whisper_indictrans2_teacher.ipynb
colab url -s nadantara-whisper --open
```

Open `/content/whisper_indictrans2_teacher.ipynb` in the browser and run only its
Phase 0 cells first. The notebook does not download a teacher until the explicit
model cell is run. Use the analogous `nadantara-seamless` session and notebook for
Teacher B. Do not assume the requested T4 is the assigned hardware; the notebook
prints the actual device and free VRAM.

Teacher B's exact setup is:

```bash
colab new -s nadantara-seamless --gpu T4
colab upload -s nadantara-seamless /tmp/nadantara-benchmark-smoke.tar.gz /content/nadantara-benchmark-smoke.tar.gz
colab upload -s nadantara-seamless experiments/teachers/seamless_m4t_teacher.ipynb /content/seamless_m4t_teacher.ipynb
colab url -s nadantara-seamless --open
```

Useful lifecycle commands:

```bash
colab status -s nadantara-whisper
colab ls -s nadantara-whisper /content
colab log -s nadantara-whisper -o /tmp/nadantara-whisper-history.jsonl
colab download -s nadantara-whisper /content/teacher_a_outputs/teacher_whisper_indictrans2.jsonl /tmp/teacher_a.jsonl
colab stop -s nadantara-whisper
```

For Teacher B, replace the session name and remote output with:

```bash
colab status -s nadantara-seamless
colab ls -s nadantara-seamless /content
colab log -s nadantara-seamless -o /tmp/nadantara-seamless-history.jsonl
colab download -s nadantara-seamless /content/teacher_b_outputs/teacher_seamless_m4t_v2.jsonl /tmp/teacher_b.jsonl
colab stop -s nadantara-seamless
```

The exact output path is printed by each notebook and should be copied into the
download command. Prefer a Drive mount from inside the notebook for long runs.

## Official model references

- Whisper: https://huggingface.co/openai/whisper-large-v3-turbo
- IndicTrans2: https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
- SeamlessM4T v2: https://huggingface.co/facebook/seamless-m4t-v2-large

IndicTrans2 requires accepting its Hugging Face access conditions and uses
`trust_remote_code=True` with a recorded model revision. SeamlessM4T weights are
CC BY-NC 4.0; retain that license in experiment metadata.
