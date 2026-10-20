# Teacher experiments

For data review and verified-transcript versus ASR translation diagnostics, run
`PYTHONPATH=src:. python experiments/teachers/audit_translation_data.py --help`.
The `prepare` command writes an audio bundle, automated checks, and a human review
CSV. After review, `translate` uses an existing local IndicTrans2 model for both
conditions; `compare` scores the exact paired set. Review is required and no model
is downloaded automatically. Write generated artifacts under ignored `artifacts/`.
Review every row: set `audio_verified` and `target_alignment` to `yes` or `no`,
enter the actual `reviewer`, and supply `verified_source_text` for accepted rows.
For paired validation, pass the existing canonical Teacher A JSONL via
`prepare --asr-outputs`; selection is restricted to successful ASR sample IDs.
Use the teacher notebook's compatible inference environment for `translate`, and
provide the exact 40-character local model source revision with `--revision`.

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

## Fixed benchmark

The first full experiment uses one deterministic 100-example validation benchmark.
Teacher A must finish before Teacher B is started. The source manifest is not
modified, and the helper verifies audio paths, writes `sample_ids.txt`, and records
SHA-256 hashes:

```bash
PYTHONPATH=. .venv/bin/python experiments/teachers/prepare_benchmark_bundle.py \
  --manifest dataset/processed/wordproject_en_hi_2k/manifest.jsonl \
  --output artifacts/teacher_benchmarks/wordproject_validation_100_seed_20260918 \
  --archive artifacts/teacher_benchmarks/wordproject_validation_100_seed_20260918.tar.gz \
  --split validation --limit 100 --seed 20260918
```

Current fixed benchmark hashes:

```text
sample_ids_sha256=f0b4f025f0aa643c66b49123387b01b354c227e382a01e848fcf848d8bee2057
manifest_sha256=77265d58aca85ee8a3d29635a5324baa1f7323cec187b49e56871c8d7d90f560
source_manifest_sha256=3b5f9ab21fe3752210548017d3f9deaeb2fbfa0c42e7fbe73cc2915946b45114
```

Keep the benchmark archive outside Git. Do not regenerate it for Teacher B.

## Interactive Colab CLI flow

The installed CLI is version 0.6.0. The browser workflow below avoids its current
`colab exec`/`jupyter-kernel-client` incompatibility and lets you run notebook cells
interactively:

```bash
colab new -s nadantara-whisper --gpu T4
colab upload -s nadantara-whisper artifacts/teacher_benchmarks/wordproject_validation_100_seed_20260918.tar.gz /content/wordproject_validation_100_seed_20260918.tar.gz
colab upload -s nadantara-whisper experiments/teachers/whisper_indictrans2_teacher.ipynb /content/whisper_indictrans2_teacher.ipynb
colab url -s nadantara-whisper --open
```

Open `/content/whisper_indictrans2_teacher.ipynb` in the browser and run only its
Phase 0 cells first. The notebook mounts Google Drive and writes all Teacher A
artifacts incrementally there. The notebook does not download a teacher until the
explicit model cell is run. Do not start Teacher B until the Teacher A stop gate is
complete and approved. Do not assume the requested T4 is the assigned hardware; the
notebook prints the actual device and free VRAM.

Teacher B's exact setup is:

```bash
colab new -s nadantara-seamless --gpu T4
colab upload -s nadantara-seamless artifacts/teacher_benchmarks/wordproject_validation_100_seed_20260918.tar.gz /content/wordproject_validation_100_seed_20260918.tar.gz
colab upload -s nadantara-seamless experiments/teachers/seamless_m4t_teacher.ipynb /content/seamless_m4t_teacher.ipynb
colab upload -s nadantara-seamless artifacts/tokenizers/hindi_2k/hindi.model /content/hindi_2k.model
colab url -s nadantara-seamless --open
```

Useful lifecycle commands:

```bash
colab status -s nadantara-whisper
colab ls -s nadantara-whisper /content
colab log -s nadantara-whisper -o /tmp/nadantara-whisper-history.jsonl
colab download -s nadantara-whisper /content/drive/MyDrive/Nadantara/teacher_a_100/teacher_a_100_artifacts.zip /tmp/teacher_a_100_artifacts.zip
colab stop -s nadantara-whisper
```

Teacher B downloads its complete persistent archive with:

```bash
colab status -s nadantara-seamless
colab ls -s nadantara-seamless /content
colab log -s nadantara-seamless -o /tmp/nadantara-seamless-history.jsonl
colab download -s nadantara-seamless /content/drive/MyDrive/Nadantara/teacher_b_100/teacher_b_100_artifacts.zip /tmp/teacher_b_100_artifacts.zip
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
