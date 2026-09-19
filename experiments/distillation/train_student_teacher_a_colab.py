"""Manual Colab entry point for Student A: Whisper -> IndicTrans2 distillation."""

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
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DATASET_CONFIG,
    DATASET_SPLIT,
    MODEL_CONFIG_RELATIVE,
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
    run_two_stage_student,
    select_and_materialize_wordproject,
    sha256_file,
    write_training_manifests,
)


MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
TRANSLATION_ID = "ai4bharat/indictrans2-en-indic-1B"
TRANSLATION_REVISION = "10e65a9951a1e922cd109a95e8aba9357b62144b"
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


def generate_whisper(rows: list[dict], shared_root: Path, output_path: Path, token: str) -> dict[str, dict]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    records = load_jsonl(output_path)
    if len(records) >= len(rows) and all(
        row.get("status") == "ok" or not RETRY_FAILED for row in records.values()
    ):
        print("whisper=complete; reusing persisted JSONL")
        return records
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=token
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        token=token,
    ).to(device)
    model.eval()
    generation_config = {
        "language": "english",
        "task": "transcribe",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 256,
    }
    print("whisper_existing_records=", len(records))
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
            "speech_teacher": MODEL_ID,
            "speech_teacher_revision": MODEL_REVISION,
            "speech_teacher_license": "MIT",
            "status": "error",
            "stage": "whisper",
            "generation_config": generation_config,
        }
        started = time.perf_counter()
        try:
            waveform, duration = _audio(shared_root / row["audio_path"])
            inputs = processor(
                waveform,
                sampling_rate=TARGET_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = {
                key: (
                    value.to(device=device, dtype=dtype)
                    if torch.is_floating_point(value)
                    else value.to(device)
                )
                for key, value in inputs.items()
            }
            forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    forced_decoder_ids=forced_ids,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=256,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            transcript = processor.batch_decode(
                generated.sequences, skip_special_tokens=True
            )[0].strip()
            mean_logprob = None
            if generated.scores:
                transition = model.compute_transition_scores(
                    generated.sequences, generated.scores, normalize_logits=True
                )
                finite = transition[0][torch.isfinite(transition[0])]
                mean_logprob = float(finite.mean().cpu()) if finite.numel() else None
            base.update(
                {
                    "status": "ok",
                    "stage": None,
                    "whisper_transcript_en": transcript,
                    "duration_seconds": duration,
                    "asr_seconds": time.perf_counter() - started,
                    "whisper_mean_logprob": mean_logprob,
                }
            )
        except Exception as exc:
            base.update({"error": repr(exc), "asr_seconds": time.perf_counter() - started})
        append_jsonl(output_path, base)
        records[sample_id] = base
        print(f"whisper={index}/{len(rows)} sample_id={sample_id} status={base['status']}")
    del model, processor
    clear_cuda()
    print(
        "whisper_summary=",
        len(records),
        "successful=",
        sum(row.get("status") == "ok" for row in records.values()),
        "failed=",
        sum(row.get("status") != "ok" for row in records.values()),
    )
    return records


def generate_indictrans(
    rows: list[dict], asr_records: dict[str, dict], output_path: Path, token: str
) -> dict[str, dict]:
    import torch
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    records = load_jsonl(output_path)
    if len(records) >= len(rows) and all(
        row.get("status") == "ok" or not RETRY_FAILED for row in records.values()
    ):
        print("indictrans=complete; reusing persisted JSONL")
        return records
    tokenizer = AutoTokenizer.from_pretrained(
        TRANSLATION_ID, revision=TRANSLATION_REVISION, trust_remote_code=True, token=token
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_ID,
        revision=TRANSLATION_REVISION,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        token=token,
    ).to("cuda")
    model.half()
    model.eval()
    processor = IndicProcessor(inference=True)
    generation_config = {
        "src_lang": "eng_Latn",
        "tgt_lang": "hin_Deva",
        "do_sample": False,
        "num_beams": 5,
        "num_return_sequences": 1,
        "max_length": 256,
        "batch_size": 1,
    }
    for index, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        if sample_id in records and (records[sample_id].get("status") == "ok" or not RETRY_FAILED):
            continue
        asr = asr_records.get(sample_id, {})
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
            "whisper_transcript_en": asr.get("whisper_transcript_en"),
            "speech_teacher": MODEL_ID,
            "speech_teacher_revision": MODEL_REVISION,
            "speech_teacher_license": "MIT",
            "translation_teacher": TRANSLATION_ID,
            "translation_teacher_revision": TRANSLATION_REVISION,
            "translation_teacher_license": "MIT",
            "teacher_pipeline": "whisper-large-v3-turbo+indictrans2-en-indic-1B",
            "status": "error",
            "stage": "indictrans2",
            "generation_config": generation_config,
        }
        started = time.perf_counter()
        if asr.get("status") != "ok":
            base.update({"stage": "whisper", "error": asr.get("error", "missing ASR output")})
        else:
            try:
                batch = processor.preprocess_batch(
                    [str(asr["whisper_transcript_en"])],
                    src_lang="eng_Latn",
                    tgt_lang="hin_Deva",
                )
                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        use_cache=True,
                        min_length=0,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                    )
                decoded = tokenizer.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                hindi = processor.postprocess_batch(decoded, lang="hin_Deva")[0]
                base.update(
                    {
                        "status": "ok",
                        "stage": None,
                        "teacher_hindi_model_decoded_raw": decoded[0],
                        "teacher_hindi_raw": hindi,
                        "teacher_hindi_normalized": normalize_hindi(hindi),
                        "translation_seconds": time.perf_counter() - started,
                        "asr_seconds": asr.get("asr_seconds"),
                    }
                )
            except Exception as exc:
                base.update({"error": repr(exc), "translation_seconds": time.perf_counter() - started})
        append_jsonl(output_path, base)
        records[sample_id] = base
        print(f"indictrans={index}/{len(rows)} sample_id={sample_id} status={base['status']}")
    del model, tokenizer, processor
    clear_cuda()
    print(
        "indictrans_summary=",
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
            "transformers==4.53.2",
            "accelerate",
            "sentencepiece",
            "IndicTransToolkit",
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
    output_root = persist_root / "student_a_whisper_indictrans"
    selection_root = shared_root / "selection"
    print("[3/8] Deterministic WordProject selection and real-data materialization")
    if (selection_root / "selected_manifest.jsonl").is_file():
        selected, selection_metadata = ensure_selected_manifest(selection_root, PSEUDO_TRAIN_EXAMPLES, True)
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

    print("[4/8] Teacher A generation: Whisper then IndicTrans2")
    asr_path = output_dataset / "teacher_a_whisper_transcripts.jsonl"
    pseudo_path = output_dataset / "teacher_a_pseudo_10k.jsonl"
    asr_records = generate_whisper(shared_selected, shared_root, asr_path, token)
    teacher_records = generate_indictrans(shared_selected, asr_records, pseudo_path, token)
    teacher_metadata = {
        "schema_version": 1,
        "teacher_pipeline": "whisper-large-v3-turbo+indictrans2-en-indic-1B",
        "speech_teacher": MODEL_ID,
        "speech_teacher_revision": MODEL_REVISION,
        "translation_teacher": TRANSLATION_ID,
        "translation_teacher_revision": TRANSLATION_REVISION,
        "teacher_licenses": {MODEL_ID: "MIT", TRANSLATION_ID: "MIT"},
        "runtime_versions": runtime_versions(),
        "generation_config": {
            "whisper": {
                "language": "english",
                "task": "transcribe",
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 256,
            },
            "indictrans2": {
                "src_lang": "eng_Latn",
                "tgt_lang": "hin_Deva",
                "do_sample": False,
                "num_beams": 5,
                "num_return_sequences": 1,
                "max_length": 256,
                "batch_size": 1,
            },
        },
        "selection_metadata": selection_metadata,
        "pseudo_output_sha256": sha256_file(pseudo_path),
        "asr_output_sha256": sha256_file(asr_path),
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
        shared_selected, teacher_records, tokenizer, output_dataset, "teacher_a", selection_metadata
    )
    if not filtered:
        raise RuntimeError("Teacher A produced no usable pseudo labels after conservative filtering")
    print("pseudo_kept=", len(filtered), "pseudo_filtered=", report["filtered_samples"])

    teacher_a_filtered_path = output_dataset / "pseudo_train_filtered.jsonl"
    teacher_b_filtered_path = (
        persist_root / "student_b_seamless_m4t/dataset/pseudo_train_filtered.jsonl"
    )
    common_ids, common_metadata = establish_common_pseudo_intersection(
        shared_root, shared_selected, teacher_a_filtered_path, teacher_b_filtered_path
    )
    if common_ids is None:
        archive_experiment(output_root)
        print(
            "STUDENT_TRAINING_STOP: Teacher B filtered manifest is missing. "
            "Run the Teacher B script, then rerun this Student A script."
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
        shared_root, output_root, filtered, real_manifest, "teacher_a"
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
    print("student_a_complete=", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
