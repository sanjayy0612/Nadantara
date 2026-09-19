"""Manual Colab entry point for Student B: SeamlessM4T-v2 distillation."""

from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import shutil
import time

import colab_common

importlib.reload(colab_common)

from colab_common import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DATASET_SPLIT,
    TARGET_RATE,
    archive_experiment,
    append_jsonl,
    bootstrap_nadantara,
    clear_cuda,
    establish_common_pseudo_intersection,
    ensure_selected_manifest,
    filter_pseudo_labels,
    get_hf_token,
    install_dependencies,
    load_jsonl,
    materialize_real_dataset,
    mount_drive,
    normalize_hindi,
    print_runtime,
    runtime_versions,
    select_and_materialize_wordproject,
    run_two_stage_student,
    sha256_file,
    write_training_manifests,
)


MODEL_ID = "facebook/seamless-m4t-v2-large"
MODEL_REVISION = "5f8cc790b19fc3f67a61c105133b20b34e3dcb76"
PSEUDO_TRAIN_EXAMPLES = int(os.environ.get("PSEUDO_TRAIN_EXAMPLES", "10000"))
RETRY_FAILED = os.environ.get("RETRY_FAILED", "0") == "1"
PSEUDO_STEPS = int(os.environ.get("PSEUDO_PRETRAIN_STEPS", "2000"))
REAL_STEPS = int(os.environ.get("REAL_FINETUNE_STEPS", "1000"))


def _audio(path: Path):
    from seamtranslation.preprocessing.audio import decode_wav

    decoded = decode_wav(path.read_bytes())
    if decoded.sample_rate != TARGET_RATE:
        raise ValueError(f"expected {TARGET_RATE} Hz audio, got {decoded.sample_rate}")
    return decoded.waveform, len(decoded.waveform) / TARGET_RATE


def _generated_ids(value, torch):
    sequences = getattr(value, "sequences", None)
    if sequences is not None:
        value = sequences
    elif isinstance(value, (tuple, list)):
        value = value[0]
    if value is None:
        raise RuntimeError("Seamless generation returned no token sequences")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().cpu()
    if value.ndim == 1:
        return value.tolist()
    if value.ndim == 2:
        return value[0].tolist()
    raise RuntimeError(f"unexpected Seamless token shape: {tuple(value.shape)}")


def generate_seamless(
    rows: list[dict], shared_root: Path, output_path: Path, token: str
) -> dict[str, dict]:
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2Model

    records = load_jsonl(output_path)
    if len(records) >= len(rows) and all(
        row.get("status") == "ok" or not RETRY_FAILED for row in records.values()
    ):
        print("seamless=complete; reusing persisted JSONL")
        return records
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=token
    )
    model = SeamlessM4Tv2Model.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        token=token,
    )
    model.eval()
    model_device = model.device
    model_dtype = next(
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ).dtype
    generation_config = {
        "src_lang": "eng",
        "tgt_lang": "hin",
        "generate_speech": False,
        "text_do_sample": False,
        "text_num_beams": 4,
        "max_new_tokens": 256,
        "batch_size": 1,
    }
    print("seamless_existing_records=", len(records))
    for index, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        if sample_id in records and (records[sample_id].get("status") == "ok" or not RETRY_FAILED):
            continue
        base = {
            "schema_version": 1,
            "sample_id": sample_id,
            "audio_path": row["audio_path"],
            "audio_sha256": row["audio_sha256"],
            "source_dataset": DATASET_ID,
            "source_dataset_revision": DATASET_REVISION,
            "source_license": DATASET_LICENSE,
            "source_config": DATASET_CONFIG,
            "source_split": DATASET_SPLIT,
            "source_text": row.get("source_text"),
            "source_asr_text": row.get("source_asr_text"),
            "reference_hindi": row.get("human_reference_hindi"),
            "teacher_pipeline": MODEL_ID,
            "speech_teacher": MODEL_ID,
            "speech_teacher_revision": MODEL_REVISION,
            "speech_teacher_license": "CC-BY-NC-4.0",
            "translation_teacher": None,
            "status": "error",
            "stage": "seamless",
            "generation_config": generation_config,
        }
        started = time.perf_counter()
        try:
            waveform, duration = _audio(shared_root / row["audio_path"])
            inputs = processor(audio=waveform, sampling_rate=TARGET_RATE, return_tensors="pt")
            inputs = {
                key: (
                    value.to(device=model_device, dtype=model_dtype)
                    if torch.is_floating_point(value)
                    else value.to(model_device)
                )
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    tgt_lang="hin",
                    generate_speech=False,
                    text_num_beams=4,
                    text_do_sample=False,
                    max_new_tokens=256,
                )
            raw_text = processor.decode(
                _generated_ids(generated, torch), skip_special_tokens=True
            ).strip()
            base.update(
                {
                    "status": "ok",
                    "stage": None,
                    "teacher_hindi_model_decoded_raw": raw_text,
                    "teacher_hindi_raw": raw_text,
                    "teacher_hindi_normalized": normalize_hindi(raw_text),
                    "translation_seconds": time.perf_counter() - started,
                    "duration_seconds": duration,
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            clear_cuda()
            base.update(
                {
                    "error_type": "cuda_oom",
                    "error": repr(exc),
                    "translation_seconds": time.perf_counter() - started,
                }
            )
        except Exception as exc:
            base.update(
                {
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                    "translation_seconds": time.perf_counter() - started,
                }
            )
        append_jsonl(output_path, base)
        records[sample_id] = base
        print(f"seamless={index}/{len(rows)} sample_id={sample_id} status={base['status']}")
    del model, processor
    clear_cuda()
    print(
        "seamless_summary=",
        len(records),
        "successful=",
        sum(row.get("status") == "ok" for row in records.values()),
        "failed=",
        sum(row.get("status") != "ok" for row in records.values()),
    )
    return records


def main() -> int:
    print("[1/8] Environment and source bootstrap")
    source_root = bootstrap_nadantara()
    install_dependencies(
        [
            "datasets>=3.5,<4",
            "transformers==5.16.1",
            "accelerate",
            "sentencepiece",
            "soundfile",
            "scipy",
            "sacrebleu",
            "huggingface_hub",
        ]
    )
    print_runtime()
    token = get_hf_token()
    print("[2/8] Google Drive persistence")
    persist_root = mount_drive()
    shared_root = persist_root / "shared_wordproject_10k"
    output_root = persist_root / "student_b_seamless_m4t"
    selection_root = shared_root / "selection"
    print("[3/8] Verify or create the deterministic shared selection and materialize real data")
    if (selection_root / "selected_manifest.jsonl").is_file():
        selected, selection_metadata = ensure_selected_manifest(
            selection_root, PSEUDO_TRAIN_EXAMPLES, require_existing=True
        )
    else:
        selected, selection_metadata = select_and_materialize_wordproject(
            source_root, selection_root, token, PSEUDO_TRAIN_EXAMPLES
        )
    shared_selected = [
        dict(row)
        | {"audio_path": str(Path("selection") / row["audio_path"])}
        for row in selected
    ]
    real_manifest = materialize_real_dataset(source_root, shared_root, token)
    output_dataset = output_root / "dataset"
    output_dataset.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selection_root / "selected_manifest.jsonl", output_dataset / "selected_manifest.jsonl")
    shutil.copy2(selection_root / "selected_train_ids.txt", output_dataset / "selected_train_ids.txt")
    shutil.copy2(selection_root / "selection_metadata.json", output_dataset / "selection_metadata.json")
    print("selected_examples=", len(shared_selected), "sample_ids_sha256=", selection_metadata["sample_ids_sha256"])

    print("[4/8] Teacher B generation: SeamlessM4T-v2 Hindi text")
    pseudo_path = output_dataset / "teacher_b_pseudo_10k.jsonl"
    teacher_records = generate_seamless(shared_selected, shared_root, pseudo_path, token)
    teacher_metadata = {
        "schema_version": 1,
        "teacher_pipeline": MODEL_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "teacher_license": "CC-BY-NC-4.0",
        "runtime_versions": runtime_versions(),
        "generation_config": {
            "src_lang": "eng",
            "tgt_lang": "hin",
            "generate_speech": False,
            "text_do_sample": False,
            "text_num_beams": 4,
            "max_new_tokens": 256,
            "batch_size": 1,
        },
        "selection_metadata": selection_metadata,
        "pseudo_output_sha256": sha256_file(pseudo_path),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dataset / "teacher_experiment_metadata.json").write_text(
        json.dumps(teacher_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[5/8] Quality flags, Hindi-2K audit, and filtered pseudo manifest")
    from seamtranslation.tokenization import HindiTokenizer

    tokenizer = HindiTokenizer(source_root / "artifacts/tokenizers/hindi_2k/hindi.model")
    filtered, report = filter_pseudo_labels(
        shared_selected, teacher_records, tokenizer, output_dataset, "teacher_b", selection_metadata
    )
    if not filtered:
        raise RuntimeError("Teacher B produced no usable pseudo labels after conservative filtering")
    print("pseudo_kept=", len(filtered), "pseudo_filtered=", report["filtered_samples"])

    teacher_a_filtered_path = (
        persist_root / "student_a_whisper_indictrans/dataset/pseudo_train_filtered.jsonl"
    )
    teacher_b_filtered_path = output_dataset / "pseudo_train_filtered.jsonl"
    common_ids, common_metadata = establish_common_pseudo_intersection(
        shared_root, shared_selected, teacher_a_filtered_path, teacher_b_filtered_path
    )
    if common_ids is None:
        archive_experiment(output_root)
        print(
            "STUDENT_TRAINING_STOP: Teacher A filtered manifest is missing. "
            "Run the Teacher A script, then rerun this Student B script."
        )
        return 0
    filtered = [row for row in filtered if row["sample_id"] in common_ids]
    if not filtered:
        raise RuntimeError("the common Teacher A/Teacher B pseudo intersection is empty")
    (output_dataset / "common_pseudo_selection.json").write_text(
        json.dumps(common_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dataset / "pseudo_train_common.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in filtered),
        encoding="utf-8",
    )
    print("common_pseudo_examples=", len(filtered), "common_sample_ids_sha256=", common_metadata["sample_ids_sha256"])

    print("[6/8] Teacher cleanup")
    clear_cuda()
    print("[7/8] Fresh 2.69M Nadantara student: pseudo pretraining then real fine-tuning")
    pseudo_manifest, real_training_manifest = write_training_manifests(
        shared_root, output_root, filtered, real_manifest, "teacher_b"
    )
    run_metadata = run_two_stage_student(
        source_root,
        shared_root,
        output_root,
        pseudo_manifest,
        real_training_manifest,
        len(filtered),
        teacher_metadata,
        pseudo_steps=PSEUDO_STEPS,
        real_steps=REAL_STEPS,
    )
    run_metadata["pseudo_quality_report_sha256"] = sha256_file(output_dataset / "pseudo_quality_report.json")
    (output_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("[8/8] Artifact export")
    archive_experiment(output_root)
    print("student_b_complete=", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
