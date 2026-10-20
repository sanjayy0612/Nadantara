"""Prepare human data review, run paired transcript translation, and score results.

No model downloads occur. IndicTrans2 inference requires an existing local model
and the same optional dependencies as the teacher notebook.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import shutil
import wave

from experiments.teachers.prepare_benchmark_bundle import sha256_file
from experiments.teachers.compare_teacher_outputs import load_jsonl, score


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    asr = load_jsonl(args.asr_outputs) if args.asr_outputs else {}
    rows = [row for row in read_rows(args.manifest) if row.get("split") == args.split]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample IDs")
    if args.asr_outputs:
        rows = [row for row in rows if asr.get(row["id"], {}).get("status") == "ok"]
    if len(rows) < args.limit:
        raise ValueError(f"only {len(rows)} eligible examples; requested {args.limit}")
    rows = sorted(rows, key=lambda row: row["id"])
    selected = sorted(random.Random(args.seed).sample(rows, args.limit), key=lambda r: r["id"])
    # Validate every audio file before creating the bundle.
    checked = []
    root = args.manifest.parent.resolve()
    for row in selected:
        audio = (root / row["audio_path"]).resolve()
        if root not in audio.parents:
            raise ValueError("audio path escapes manifest root")
        with wave.open(str(audio)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError(f"expected mono 16-bit 16 kHz PCM: {audio}")
            duration = wav.getnframes() / wav.getframerate()
        checked.append((row, audio, duration, sha256_file(audio)))
    args.output.mkdir(parents=True)
    (args.output / "audio").mkdir()
    tokenizer = None
    if args.tokenizer:
        from seamtranslation.tokenization import HindiTokenizer
        tokenizer = HindiTokenizer(args.tokenizer)
    review, bundle = [], []
    for index, (row, audio, duration, digest) in enumerate(checked):
        relative_audio = f"audio/{index:03d}.wav"
        shutil.copyfile(audio, args.output / relative_audio)
        target = str(row.get("target_text", ""))
        source = str(row.get("source_text", ""))
        record = {
            "sample_id": row["id"], "audio_path": relative_audio,
            "audio_sha256": digest, "duration_seconds": duration,
            "source_text": source, "reference_hindi": target,
            "asr_transcript": asr.get(row["id"], {}).get("whisper_transcript_en", ""),
            "source_split": row["split"], "dataset_id": row.get("dataset_id"),
            "dataset_revision": row.get("dataset_revision"),
            "source_license": row.get("source_license"),
            "target_tokens": len(tokenizer.encode(target)) if tokenizer else None,
        }
        bundle.append(record)
        review.append({"sample_id": row["id"], "audio_verified": "pending",
                       "target_alignment": "pending", "verified_source_text": "",
                       "reviewer": "", "notes": ""})
    manifest = args.output / "audit_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bundle))
    with (args.output / "review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    write_json(args.output / "checks.json", {
        "source_manifest_sha256": sha256_file(args.manifest),
        "audit_manifest_sha256": sha256_file(manifest), "seed": args.seed,
        "split": args.split, "selected": len(bundle), "human_review": "pending",
        "asr_outputs_sha256": sha256_file(args.asr_outputs) if args.asr_outputs else None,
        "tokenizer_sha256": sha256_file(args.tokenizer) if args.tokenizer else None,
        "empty_source_ids": [r["sample_id"] for r in bundle if not r["source_text"].strip()],
        "empty_target_ids": [r["sample_id"] for r in bundle if not r["reference_hindi"].strip()],
        "target_truncation_ids": [r["sample_id"] for r in bundle
                                  if r["target_tokens"] is not None and r["target_tokens"] > 127],
        "over_product_15_second_limit_ids": [r["sample_id"] for r in bundle
                                              if r["duration_seconds"] > 15],
    })
    print(f"Prepared {len(bundle)} examples at {args.output}; human review is pending.")


def reviewed_rows(bundle):
    rows = read_rows(bundle / "audit_manifest.jsonl")
    checks = json.loads((bundle / "checks.json").read_text())
    if checks["audit_manifest_sha256"] != sha256_file(bundle / "audit_manifest.jsonl"):
        raise ValueError("audit manifest changed since preparation")
    with (bundle / "review.csv").open(newline="") as handle:
        reviews = list(csv.DictReader(handle))
    review_by_id = {row["sample_id"]: row for row in reviews}
    if len(reviews) != len(review_by_id) or set(review_by_id) != {r["sample_id"] for r in rows}:
        raise ValueError("review IDs must match bundle IDs exactly, without duplicates")
    accepted = []
    for row in rows:
        review = review_by_id[row["sample_id"]]
        if any(review[key] not in {"yes", "no"} for key in ("audio_verified", "target_alignment")):
            raise ValueError("finish human review: flags must be yes or no")
        if not review["reviewer"].strip():
            raise ValueError("reviewer is required for each decision")
        if review["audio_verified"] == review["target_alignment"] == "yes":
            if not review["verified_source_text"].strip() or not row["reference_hindi"].strip():
                raise ValueError("accepted rows need a verified transcript and Hindi reference")
            if sha256_file(bundle / row["audio_path"]) != row["audio_sha256"]:
                raise ValueError("reviewed audio hash differs")
            accepted.append({**row, "verified_source_text": review["verified_source_text"]})
    if not accepted:
        raise ValueError("no accepted examples")
    return accepted, len(rows)


def translate(args):
    rows, requested = reviewed_rows(args.bundle)
    if any(not row["asr_transcript"].strip() for row in rows):
        raise ValueError("paired test needs ASR transcripts for every accepted example")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.model_dir.is_dir():
        raise ValueError("provide a local IndicTrans2 model directory; no automatic downloads")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("revision must be an exact 40-character model commit SHA")
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                               trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True,
        attn_implementation="eager").to(args.device).eval()
    processor = IndicProcessor(inference=True)
    config = {"do_sample": False, "num_beams": 5, "max_length": 256}
    args.output.mkdir(parents=True)
    write_json(args.output / "provenance.json", {
        "model_dir": str(args.model_dir.resolve()), "revision": args.revision,
        "revision_verification": "user-provided source revision; local file hashes retained",
        "model_artifact_sha256": {
            str(path.relative_to(args.model_dir)): sha256_file(path)
            for path in sorted(args.model_dir.rglob("*")) if path.is_file()
            and path.suffix in {".safetensors", ".bin", ".json", ".model", ".py"}
        },
        "model_config_sha256": sha256_file(args.model_dir / "config.json"),
        "review_sha256": sha256_file(args.bundle / "review.csv"),
        "audit_manifest_sha256": sha256_file(args.bundle / "audit_manifest.jsonl"),
        "requested_examples": requested, "accepted_examples": len(rows),
        "generation_config": config, "device": args.device,
        "torch_version": torch.__version__,
    })
    with (args.output / "predictions.jsonl").open("w") as handle:
        for row in rows:
            for condition, field in (("verified", "verified_source_text"), ("asr", "asr_transcript")):
                texts = processor.preprocess_batch([row[field]], src_lang="eng_Latn", tgt_lang="hin_Deva")
                inputs = tokenizer(texts, truncation=False, return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, **config)
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=True)
                prediction = processor.postprocess_batch(decoded, lang="hin_Deva")[0]
                record = {"sample_id": row["sample_id"], "condition": condition,
                          "status": "ok", "source_text": row[field], "teacher_hindi_raw": prediction}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(row["sample_id"], condition, flush=True)


def compare(args):
    rows, requested = reviewed_rows(args.bundle)
    records = read_rows(args.predictions)
    lookup = {}
    for record in records:
        key = (record["sample_id"], record["condition"])
        if key in lookup or record["condition"] not in {"verified", "asr"}:
            raise ValueError("duplicate prediction or invalid condition")
        lookup[key] = record
    expected = {(row["sample_id"], condition) for row in rows for condition in ("verified", "asr")}
    if set(lookup) != expected:
        raise ValueError("predictions must cover exactly both conditions for every accepted example")
    scored = {}
    for condition, field in (("verified", "verified_source_text"), ("asr", "asr_transcript")):
        paired = []
        for row in rows:
            prediction = lookup[row["sample_id"], condition]
            if prediction.get("status") != "ok" or not prediction.get("teacher_hindi_raw", "").strip():
                raise ValueError("failed or empty predictions; do not silently score a filtered subset")
            if not row[field].strip() or prediction.get("source_text") != row[field]:
                raise ValueError("prediction source differs from reviewed input")
            paired.append({**prediction, "reference_hindi": row["reference_hindi"]})
        scored[condition] = score(paired)
    report = {"requested_examples": requested, "accepted_examples": len(rows),
              "excluded_examples": requested - len(rows), "conditions": scored,
              "verified_minus_asr": {metric: scored["verified"][metric] - scored["asr"][metric]
                                      for metric in ("chrf2", "bleu")},
              "review_sha256": sha256_file(args.bundle / "review.csv"),
              "predictions_sha256": sha256_file(args.predictions),
              "limitations": "Human review required. Development diagnostic; no universal pass threshold. "
                             "Both conditions must use the same model and generation settings."}
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--asr-outputs", type=Path)
    prep.add_argument("--tokenizer", type=Path)
    prep.add_argument("--split", choices=["train", "validation"], default="train")
    prep.add_argument("--limit", type=int, default=30)
    prep.add_argument("--seed", type=int, default=17)
    prep.set_defaults(run=prepare)
    run = commands.add_parser("translate")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--revision", required=True, help="Pinned local IndicTrans2 source revision")
    run.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(run=translate)
    scoring = commands.add_parser("compare")
    scoring.add_argument("--bundle", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
