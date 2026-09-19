# First Nadantara distillation experiment

These are manual Google Colab entry scripts. They do not create runtimes or use the
Colab CLI. The same entry scripts can also run on a local CUDA workstation; outside
Colab, `NADANTARA_DRIVE_ROOT` means a local copy or synced copy of the artifact folder.

## Local preparation

Create the small source bundle once:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/distillation/prepare_colab_source_bundle.py \
  --output /tmp/nadantara_source.tar.gz
```

Upload these files to a clean Colab session:

- `train_student_teacher_a_colab.py` or `train_student_teacher_b_colab.py`
- `colab_common.py`
- `/tmp/nadantara_source.tar.gz`

The source bundle contains only Nadantara Python code, the exact 2K tokenizer, the
baseline 2K manifest, the baseline semantic config, and the held-out benchmark ID
guard/manifest. It does not contain model weights or the 526 MB local audio subset.

Add a Colab Secret named `HF_TOKEN`. The script reads it through Colab Secret access
and never embeds it in source.

If Teacher A and Teacher B run under different Gmail accounts, share the entire
`MyDrive/Nadantara/distillation` folder from Account A with Account B as Editor. In
Account B, use Google Drive's **Organize → Add shortcut** to place that shared folder
inside Account B's My Drive at the same path. Both sessions must then use the same
Drive root:

```python
import os
os.environ["NADANTARA_DRIVE_ROOT"] = "/content/drive/MyDrive/Nadantara/distillation"
```

Set this before `%run`. If the shortcut has a different path, set the variable to
that exact path. The scripts refuse to create a new root when this variable is set,
which prevents Account B from silently writing to a different experiment directory.

Run the scripts in this order. This order is deliberate:

1. Run Student A. It creates the one shared 10k selection, generates Teacher A labels,
   writes its quality-filtered manifest, then stops before student training because
   Teacher B's filtered manifest does not exist yet.
2. Run Student B. It reuses and verifies the exact shared selection (or deterministically
   creates that same selection if started independently), generates Teacher B labels,
   writes its filtered manifest, creates the common usable-ID intersection, and trains
   Student B.
3. Rerun Student A. It reuses both persisted teacher outputs and trains Student A on
   the same common usable-ID intersection.

This extra rerun is a fairness gate: teacher-specific failures can differ, so neither
student is allowed to train until both filtered manifests exist. Student B requires
the Drive-persisted selection created by Student A when available, and verifies its
sample-ID hash instead of selecting a different dataset. With the shared Drive folder,
Account B sees Student A's filtered manifest and the common-intersection gate works
without copying the 10,000 audio files.

Optional environment variables, set before running the script:

```text
PSEUDO_TRAIN_EXAMPLES=10000
PSEUDO_PRETRAIN_STEPS=2000
REAL_FINETUNE_STEPS=1000
RETRY_FAILED=1
VERIFY_SHARED_AUDIO_HASHES=1
```

The default 2,000 pseudo-pretraining updates plus 1,000 real fine-tuning updates keep
the initial comparison at 3,000 optimizer updates, while ensuring the real data is the
final supervision stage. Both students use seed 17 and the unchanged 2.69M model
configuration.

## Drive layout

The scripts persist under `MyDrive/Nadantara/distillation/`:

```text
shared_wordproject_10k/
  selection/
    selected_manifest.jsonl
    selected_train_ids.txt
    selection_metadata.json
    audio/
  real_data/
    manifest.jsonl
    audio/
  training_manifests/

student_a_whisper_indictrans/
student_b_seamless_m4t/

shared_wordproject_10k/common_pseudo_selection.json
```

Teacher generation appends one JSON object per sample to Drive. Rerunning a script
skips completed successful records and, unless `RETRY_FAILED=1`, preserves failures for
inspection. The pseudo quality report records every sample and every exclusion reason.

The scripts do not use the benchmark or its human Hindi text as pseudo supervision.
WordProject `hi_text` is retained only as a diagnostic reference; `target_text` in the
student pseudo manifest comes from the selected teacher output.

The first run can stop intentionally with `STUDENT_TRAINING_STOP`. That is expected
until both teacher-specific `pseudo_train_filtered.jsonl` files are present. Do not
interpret that stop as a model failure.

By default, later account/session reruns verify the shared manifest, IDs, and audio
file existence without re-reading 10,000 Drive files to hash them again. Set
`VERIFY_SHARED_AUDIO_HASHES=1` for a full integrity pass when needed; it is expected
to be slow on Google Drive-mounted storage.

## Continue on a local RTX 5050 workstation

Colab's "Cannot connect to GPU backend" message is a Colab usage-limit failure. It
does not invalidate the labels already written to Drive. The Teacher B JSONL is
written after each sample, so a local run can resume from the last persisted record.

Download or sync the entire `Nadantara/distillation` folder from Drive to the laptop.
Do not download only the final ZIP if Teacher B is still running: the partial file
needed for resume is:

```text
distillation/
  shared_wordproject_10k/
    selection/                 # selected_manifest.jsonl plus all audio/
    real_data/                 # existing real train/validation audio
  student_a_whisper_indictrans/
    dataset/pseudo_train_filtered.jsonl
  student_b_seamless_m4t/
    dataset/teacher_b_pseudo_10k.jsonl   # partial or complete resume state
```

The `selection/audio` and `real_data/audio` directories are required for the
remaining teacher generation and student training. A local SSD copy is preferable
to a cloud-synced placeholder because repeated audio reads are much faster locally.

On the laptop, use the repository checkout as the source code and set the artifact
folder explicitly. For macOS/Linux:

```bash
cd /path/to/seamTranslation
export HF_TOKEN="hf_..."
export NADANTARA_DRIVE_ROOT="/path/to/local/distillation"
export NADANTARA_SOURCE_ROOT="/path/to/seamTranslation"
python experiments/distillation/train_student_teacher_b_colab.py
```

For PowerShell:

```powershell
$env:HF_TOKEN = "hf_..."
$env:NADANTARA_DRIVE_ROOT = "D:\Nadantara\distillation"
$env:NADANTARA_SOURCE_ROOT = "D:\Nadantara\seamTranslation"
python experiments\distillation\train_student_teacher_b_colab.py
```

Before the expensive run, verify that the local PyTorch installation sees the
5050 GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
```

If this prints `False`, install a CUDA-enabled PyTorch build appropriate for that
laptop before running the distillation script. The script installs the remaining
Python dependencies, but it does not replace the system's CUDA-enabled PyTorch.

The local script will print the same 10k sample-ID hash as the shared selection. If
the hash differs, stop; the wrong artifact folder was selected. If it matches, it
will reuse completed records in `teacher_b_pseudo_10k.jsonl`, generate only missing
Teacher B labels, then continue to filtering and Student B training.

After Student B finishes, run the Teacher A script locally with the same two
environment paths:

```bash
python experiments/distillation/train_student_teacher_a_colab.py
```

Teacher A already has its labels, so this rerun should skip generation and train
Student A after the common Teacher A/Teacher B usable-ID intersection is available.
The local results are written to the local artifact folder. Copy or sync that folder
back to Drive afterward if it needs to be preserved there.

Teacher model weights are downloaded into the normal local Hugging Face cache; they
are not expected inside the Drive artifact folder and do not need to be copied from
Colab.
