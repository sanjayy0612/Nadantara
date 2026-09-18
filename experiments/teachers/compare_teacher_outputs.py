"""Compare canonical Teacher A and Teacher B JSONL outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata
import zlib

import sacrebleu


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def diagnostics(text: str, source: str) -> dict[str, float | int | bool]:
    words = text.split()
    repeated = sum(a == b for a, b in zip(words, words[1:]))
    transitions = max(0, len(words) - 1)
    devanagari = sum("\u0900" <= char <= "\u097f" for char in text)
    letters = sum(char.isalpha() for char in text)
    source_norm = normalize_text(source).casefold()
    text_norm = normalize_text(text).casefold()
    latin_ratio = sum("a" <= char.lower() <= "z" for char in text) / max(1, letters)
    return {
        "empty": not text_norm,
        "mostly_latin": bool(letters) and latin_ratio > 0.8,
        "devanagari_ratio": devanagari / max(1, len(text)),
        "adjacent_word_repetition_rate": repeated / transitions if transitions else 0.0,
        "source_copy": bool(text_norm) and text_norm == source_norm,
        "word_count": len(words),
        "char_count": len(text),
        "compression_ratio": len(text.encode("utf-8")) / max(1, len(zlib.compress(text.encode("utf-8")))),
    }


def load_jsonl(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {path} line {line_number}")
            rows[sample_id] = row
    return rows


def score(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "ok"]
    hypotheses = [normalize_text(str(row.get("teacher_hindi_raw", row.get("teacher_hindi", "")))) for row in successful]
    references = [normalize_text(str(row.get("reference_hindi", ""))) for row in successful]
    if successful:
        per_row = [diagnostics(text, str(row.get("source_text", ""))) for text, row in zip(hypotheses, successful, strict=True)]
        metrics: dict[str, object] = {
            "bleu": float(sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl", use_effective_order=True).score),
            "chrf2": float(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score),
            "empty_output_count": sum(bool(item["empty"]) for item in per_row),
            "exact_match_count": sum(a == b for a, b in zip(hypotheses, references, strict=True)),
            "adjacent_repetition_rate": sum(float(item["adjacent_word_repetition_rate"]) for item in per_row) / len(per_row),
            "mostly_latin_count": sum(bool(item["mostly_latin"]) for item in per_row),
            "source_copy_count": sum(bool(item["source_copy"]) for item in per_row),
            "mean_devanagari_ratio": sum(float(item["devanagari_ratio"]) for item in per_row) / len(per_row),
        }
    else:
        metrics = {
            "bleu": None,
            "chrf2": None,
            "empty_output_count": 0,
            "exact_match_count": 0,
            "adjacent_repetition_rate": None,
            "mostly_latin_count": 0,
            "source_copy_count": 0,
            "mean_devanagari_ratio": None,
        }
    total_seconds = sum(float(row.get("translation_seconds", 0.0) or 0.0) for row in rows)
    failures = Counter(str(row.get("stage", "unknown")) for row in rows if row.get("status") != "ok")
    return {
        "requested_samples": len(rows),
        "successful_samples": len(successful),
        "failed_samples": len(rows) - len(successful),
        "failure_stages": dict(sorted(failures.items())),
        "total_inference_seconds": total_seconds,
        "average_inference_seconds": total_seconds / len(rows) if rows else None,
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-a", type=Path, required=True)
    parser.add_argument("--teacher-b", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()
    teacher_a = load_jsonl(args.teacher_a)
    teacher_b = load_jsonl(args.teacher_b)
    common = sorted(set(teacher_a) & set(teacher_b))
    common_success = [
        sample_id
        for sample_id in common
        if teacher_a[sample_id].get("status") == "ok"
        and teacher_b[sample_id].get("status") == "ok"
    ]
    report: dict[str, object] = {
        "teacher_a_all": score([teacher_a[sample_id] for sample_id in sorted(teacher_a)]),
        "teacher_b_all": score([teacher_b[sample_id] for sample_id in sorted(teacher_b)]),
        "common_successful_intersection": {
            "sample_count": len(common_success),
            "teacher_a": score([teacher_a[sample_id] for sample_id in common_success]),
            "teacher_b": score([teacher_b[sample_id] for sample_id in common_success]),
        },
        "manual_examples": [],
    }
    for sample_id in common[: args.examples]:
        row_a = teacher_a[sample_id]
        row_b = teacher_b[sample_id]
        report["manual_examples"].append(
            {
                "sample_id": sample_id,
                "source_text": row_a.get("source_text", row_b.get("source_text")),
                "whisper_transcript_en": row_a.get("whisper_transcript_en"),
                "reference_hindi": row_a.get("reference_hindi", row_b.get("reference_hindi")),
                "teacher_a_hindi": row_a.get("teacher_hindi_raw"),
                "teacher_b_hindi": row_b.get("teacher_hindi_raw"),
                "teacher_a_status": row_a.get("status"),
                "teacher_b_status": row_b.get("status"),
            }
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
