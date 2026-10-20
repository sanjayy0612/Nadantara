import argparse
import csv
import json
import wave

import pytest

from experiments.teachers.audit_translation_data import compare, prepare, reviewed_rows


@pytest.fixture
def audit_bundle(tmp_path):
    rows = []
    for index in range(3):
        audio = tmp_path / f"{index}.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            handle.writeframes(b"\0\0" * 160)
        rows.append({"id": str(index), "audio_path": audio.name, "split": "validation",
                     "source_text": "This is a test", "target_text": "यह एक परीक्षण है"})
    manifest = tmp_path / "source.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    asr = tmp_path / "asr.jsonl"
    asr.write_text("".join(json.dumps({"sample_id": str(i), "status": "ok",
                                     "whisper_transcript_en": "This is test"}) + "\n"
                           for i in range(3)))
    output = tmp_path / "bundle"
    prepare(argparse.Namespace(manifest=manifest, output=output, asr_outputs=asr,
                               tokenizer=None, split="validation", limit=2, seed=17))
    return output


def complete_review(bundle):
    path = bundle / "review.csv"
    with path.open(newline="") as handle:
        reviews = list(csv.DictReader(handle))
    for row in reviews:
        row.update(audio_verified="yes", target_alignment="yes",
                   verified_source_text="This is a test", reviewer="test fixture")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)


def test_prepare_preserves_audio_and_requires_human_review(audit_bundle):
    checks = json.loads((audit_bundle / "checks.json").read_text())
    assert checks["selected"] == 2
    assert checks["human_review"] == "pending"
    assert len(list((audit_bundle / "audio").glob("*.wav"))) == 2
    with pytest.raises(ValueError, match="finish human review"):
        reviewed_rows(audit_bundle)


def test_review_rejects_modified_audio(audit_bundle):
    complete_review(audit_bundle)
    (audit_bundle / "audio/000.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="audio hash"):
        reviewed_rows(audit_bundle)


def test_compare_uses_exact_paired_inputs_and_references(audit_bundle):
    complete_review(audit_bundle)
    rows, _ = reviewed_rows(audit_bundle)
    records = [{"sample_id": row["sample_id"], "condition": condition, "status": "ok",
                "source_text": row[field], "teacher_hindi_raw": row["reference_hindi"]}
               for row in rows for condition, field in
               (("verified", "verified_source_text"), ("asr", "asr_transcript"))]
    predictions = audit_bundle / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in records))
    output = audit_bundle / "results.json"
    args = argparse.Namespace(bundle=audit_bundle, predictions=predictions, output=output)
    compare(args)
    report = json.loads(output.read_text())
    assert report["accepted_examples"] == 2
    assert report["conditions"]["verified"]["chrf2"] == 100
    assert report["verified_minus_asr"]["chrf2"] == 0
    records[0]["source_text"] = "wrong input"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="source differs"):
        compare(args)
    predictions.write_text("".join(json.dumps(row) + "\n" for row in records[1:]))
    with pytest.raises(ValueError, match="cover exactly"):
        compare(args)
