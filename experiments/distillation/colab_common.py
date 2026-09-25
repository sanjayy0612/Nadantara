"""Shared, dependency-light helpers for the two manual Colab distillation runs."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import unicodedata


DATASET_ID = "ai4bharat/WordProject"
DATASET_CONFIG = "en2indic"
DATASET_SPLIT = "en2indic"
DATASET_REVISION = "82398593980643a685816a77236adaa05591b12d"
DATASET_LICENSE = "CC-BY-4.0"
EXPECTED_REAL_MANIFEST_SHA256 = "3b5f9ab21fe3752210548017d3f9deaeb2fbfa0c42e7fbe73cc2915946b45114"
TOKENIZER_SHA256 = "34b1c9c8f24b9fdb27a393978a0034fd96ea35b835b1a667e4373098a483137d"
TOKENIZER_RELATIVE = Path("artifacts/tokenizers/hindi_2k/hindi.model")
REAL_MANIFEST_RELATIVE = Path("dataset/processed/wordproject_en_hi_2k/manifest.jsonl")
BENCHMARK_MANIFEST_RELATIVE = Path(
    "artifacts/teacher_benchmarks/wordproject_validation_100_seed_20260918/benchmark_manifest.jsonl"
)
MODEL_CONFIG_RELATIVE = Path("train/configs/semantic-2k-m2.json")
TARGET_RATE = 16_000
MAX_SEMANTIC_TOKENS = 128
SELECTION_SEED = 20260918
BENCHMARK_SAMPLE_IDS_SHA256 = "f0b4f025f0aa643c66b49123387b01b354c227e382a01e848fcf848d8bee2057"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def normalize_hindi(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {path}:{line_number}")
            rows[sample_id] = row
    return rows


def clear_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_runtime() -> None:
    import platform
    import torch

    print("python=", platform.python_version())
    print("torch=", torch.__version__)
    print("cuda=", torch.version.cuda)
    print("cuda_available=", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required. On Colab, select a GPU runtime; locally, install a "
            "CUDA-enabled PyTorch build and verify that NVIDIA drivers are available."
        )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print("gpu=", props.name)
    print("gpu_vram_gib=", round(props.total_memory / 1024**3, 3))
    print("gpu_free_gib=", round(free_bytes / 1024**3, 3))
    print("gpu_total_gib=", round(total_bytes / 1024**3, 3))


def runtime_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    import torch

    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    for package in ("transformers", "datasets", "accelerate", "IndicTransToolkit"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def get_hf_token() -> str:
    value = os.environ.get("HF_TOKEN")
    if not value:
        try:
            from google.colab import userdata

            value = userdata.get("HF_TOKEN")
        except Exception:
            value = None
    if not value:
        raise RuntimeError(
            "HF_TOKEN is missing. In Colab, add a Secret named HF_TOKEN; locally, set "
            "the HF_TOKEN environment variable. It needs access to ai4bharat/WordProject "
            "and the teacher model repositories."
        )
    os.environ["HF_TOKEN"] = str(value)
    return str(value)


def mount_drive() -> Path:
    configured_root = os.environ.get("NADANTARA_DRIVE_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                "NADANTARA_DRIVE_ROOT does not exist. Set it to the downloaded or locally "
                f"synced distillation folder: {root}"
            )
        print("artifact_root=", root)
        return root

    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "Outside Colab, set NADANTARA_DRIVE_ROOT to the local copy of the "
            "Nadantara/distillation folder before running this script."
        ) from exc
    drive.mount("/content/drive", force_remount=False)
    root = Path("/content/drive/MyDrive/Nadantara/distillation")
    root.mkdir(parents=True, exist_ok=True)
    print("artifact_root=", root.resolve())
    return root


def bootstrap_nadantara() -> Path:
    """Unpack or locate a small source bundle without cloning a private repository."""

    configured_root = os.environ.get("NADANTARA_SOURCE_ROOT")
    candidates = []
    if configured_root:
        candidates.append(Path(configured_root).expanduser())
    candidates.extend([Path("/content/Nadantara"), Path.cwd() / "Nadantara", Path.cwd()])
    root = next(
        (candidate for candidate in candidates if (candidate / "src/seamtranslation").is_dir()),
        None,
    )
    archive_value = os.environ.get("NADANTARA_SOURCE_ARCHIVE")
    archive_candidates = []
    if archive_value:
        archive_candidates.append(Path(archive_value).expanduser())
    archive_candidates.extend(
        [Path("/content/nadantara_source.tar.gz"), Path.cwd() / "nadantara_source.tar.gz"]
    )
    archive = next((candidate for candidate in archive_candidates if candidate.is_file()), None)
    if root is None and archive is not None and archive.is_file():
        destination = Path(os.environ.get("NADANTARA_SOURCE_EXTRACT_ROOT", tempfile.gettempdir()))
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if not str(target).startswith(str(destination.resolve()) + os.sep):
                    raise RuntimeError(f"unsafe source archive member: {member.name}")
            bundle.extractall(destination)
        root = destination / "Nadantara"
    if root is None or not (root / "src/seamtranslation").is_dir():
        raise FileNotFoundError(
            "Nadantara source is missing. Run this from the repository root, set "
            "NADANTARA_SOURCE_ROOT, or provide nadantara_source.tar.gz via "
            "NADANTARA_SOURCE_ARCHIVE."
        )
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    return root


def install_dependencies(packages: list[str]) -> None:
    import subprocess

    marker_root = Path(os.environ.get("NADANTARA_CACHE_ROOT", tempfile.gettempdir()))
    marker_root.mkdir(parents=True, exist_ok=True)
    marker = marker_root / (".nadantara_distillation_deps_" + sha256_text("\n".join(packages))[:12])
    if marker.exists():
        return
    command = [sys.executable, "-m", "pip", "install", "-q", *packages]
    print("installing=", " ".join(packages))
    subprocess.run(command, check=True)
    marker.write_text("ok\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def benchmark_ids(source_root: Path) -> set[str]:
    path = source_root / "experiments/distillation/held_out_benchmark_ids.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    observed_hash = sha256_text("\n".join(sorted(ids)) + "\n")
    # The benchmark artifact hashes the file's original line order, which is sorted by construction.
    if observed_hash != BENCHMARK_SAMPLE_IDS_SHA256:
        raise RuntimeError(
            "held_out_benchmark_ids.txt does not match the immutable 100-example benchmark "
            f"hash: {observed_hash}"
        )
    return ids


def existing_real_guards(source_root: Path) -> dict[str, set[str]]:
    manifest = source_root / REAL_MANIFEST_RELATIVE
    if sha256_file(manifest) != EXPECTED_REAL_MANIFEST_SHA256:
        raise RuntimeError("the 2K real manifest hash differs from the baseline manifest")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    by_id = {str(row["id"]): row for row in rows}
    benchmark_manifest = source_root / BENCHMARK_MANIFEST_RELATIVE
    if not benchmark_manifest.is_file():
        raise FileNotFoundError(
            f"benchmark manifest is missing from the source bundle: {benchmark_manifest}"
        )
    benchmark_ids_in_real = set()
    benchmark_content_sha256 = set()
    benchmark_rows = [
        json.loads(line)
        for line in benchmark_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    benchmark_ids_in_real = {
        str(row["sample_id"])
        for row in benchmark_rows
        if str(row["sample_id"]) in by_id
    }
    if len(benchmark_ids_in_real) != len(benchmark_rows):
        raise RuntimeError(
            "the held-out benchmark manifest is not fully represented by the pinned real manifest"
        )
    benchmark_content_sha256 = {str(row["audio_sha256"]) for row in benchmark_rows}
    protected_rows = [by_id[sample_id] for sample_id in benchmark_ids_in_real]
    return {
        "ids": {str(row["id"]) for row in rows},
        "content_sha256": {str(row["content_sha256"]) for row in rows} | benchmark_content_sha256,
        "original_audio_sha256": {str(row["original_audio_sha256"]) for row in rows},
        "recording_hash": {str(row["recording_hash"]) for row in rows}
        | {str(row["recording_hash"]) for row in protected_rows},
    }


def _audio_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), bytes):
            return value["bytes"]
        path = value.get("path")
        if path:
            return Path(str(path)).read_bytes()
    raise TypeError(f"unsupported WordProject audio value: {type(value)!r}")


def stream_wordproject(token: str):
    from datasets import Audio, load_dataset

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
        token=token,
        revision=DATASET_REVISION,
    )
    return dataset.cast_column("chunked_audio_filepath", Audio(decode=False))


def _recording_split(recording_id: str, validation_fraction: float = 0.1) -> str:
    bucket = int(sha256_text(recording_id)[:16], 16) / 2**64
    return "validation" if bucket < validation_fraction else "train"


def _ngram_repeat_ratio(words: list[str], n: int) -> float:
    grams = [tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def quality_flags(text: str, source_text: str, tokenizer) -> dict:
    normalized = normalize_hindi(text)
    words = normalized.split()
    letters = sum(character.isalpha() for character in normalized)
    latin_letters = sum("a" <= character.lower() <= "z" for character in normalized)
    devanagari = sum("\u0900" <= character <= "\u097f" for character in normalized)
    token_ids = tokenizer.encode(normalized)
    source_norm = normalize_hindi(source_text).casefold()
    text_norm = normalized.casefold()
    source_words = source_norm.split()
    target_words = text_norm.split()
    source_set = set(source_words)
    overlap = sum(word in source_set for word in target_words) / max(1, len(target_words))
    return {
        "empty": not normalized,
        "mostly_latin": bool(letters) and latin_letters / letters > 0.8,
        "latin_letter_ratio": latin_letters / max(1, letters),
        "devanagari_ratio": devanagari / max(1, len(normalized)),
        "source_copy_exact": bool(normalized) and text_norm == source_norm,
        "source_token_overlap": overlap,
        "word_count": len(words),
        "char_count": len(normalized),
        "target_source_word_ratio": len(target_words) / max(1, len(source_words)),
        "token_count": len(token_ids),
        "unknown_token_count": sum(token == tokenizer.special_ids.unknown for token in token_ids),
        "exceeds_max_semantic_tokens": len(token_ids) + 1 > MAX_SEMANTIC_TOKENS,
        "repeated_unigram_ratio": _ngram_repeat_ratio(words, 1),
        "repeated_bigram_ratio": _ngram_repeat_ratio(words, 2),
        "repeated_trigram_ratio": _ngram_repeat_ratio(words, 3),
        "repeated_4gram_ratio": _ngram_repeat_ratio(words, 4),
    }


def filter_reason(flags: dict) -> str | None:
    """Conservative first-pass exclusions; all flags remain in the quality report."""

    if flags["empty"]:
        return "empty_output"
    if flags["mostly_latin"]:
        return "mostly_latin_output"
    if flags["exceeds_max_semantic_tokens"]:
        return "semantic_length_exceeds_128"
    if (
        flags["token_count"] >= 16
        and (
            flags["repeated_trigram_ratio"] >= 0.8
            or flags["repeated_4gram_ratio"] >= 0.8
        )
    ):
        return "severe_ngram_repetition"
    return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def filter_pseudo_labels(
    selected_rows: list[dict],
    records: dict[str, dict],
    tokenizer,
    output_dir: Path,
    teacher_tag: str,
    selection_metadata: dict,
) -> tuple[list[dict], dict]:
    """Audit every teacher record and write a conservative student manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered: list[dict] = []
    per_sample: list[dict] = []
    reason_counts: Counter[str] = Counter()
    successful = 0
    for source_row in selected_rows:
        sample_id = str(source_row["sample_id"])
        record = records.get(sample_id)
        if record is None or record.get("status") != "ok":
            reason = "teacher_failure"
            flags = {"status": record.get("status") if record else "missing"}
            reason_counts[reason] += 1
            per_sample.append({"sample_id": sample_id, "reason": reason, "flags": flags})
            continue
        successful += 1
        raw = str(record.get("teacher_hindi_raw", ""))
        flags = quality_flags(raw, str(source_row.get("source_text", "")), tokenizer)
        reason = filter_reason(flags)
        reason_counts[reason or "kept"] += 1
        per_sample.append({"sample_id": sample_id, "reason": reason, "flags": flags})
        if reason is None:
            filtered.append(
                dict(source_row)
                | {
                    "teacher_hindi_raw": raw,
                    "teacher_hindi_normalized": normalize_hindi(
                        str(record.get("teacher_hindi_normalized", raw))
                    ),
                    "teacher_pipeline": record.get("teacher_pipeline"),
                    "speech_teacher": record.get("speech_teacher"),
                    "speech_teacher_revision": record.get("speech_teacher_revision"),
                    "translation_teacher": record.get("translation_teacher"),
                    "translation_teacher_revision": record.get("translation_teacher_revision"),
                    "generation_config": record.get("generation_config"),
                    "label_origin": "pseudo",
                    "quality_flags": flags,
                }
            )
    token_counts = [item["flags"]["token_count"] for item in per_sample if "token_count" in item["flags"]]
    report = {
        "schema_version": 1,
        "teacher_tag": teacher_tag,
        "requested_samples": len(selected_rows),
        "successful_teacher_outputs": successful,
        "kept_samples": len(filtered),
        "filtered_samples": len(selected_rows) - len(filtered),
        "filter_reason_counts": dict(sorted(reason_counts.items())),
        "filter_policy": {
            "empty_output": True,
            "mostly_latin_threshold": 0.8,
            "semantic_length_limit": MAX_SEMANTIC_TOKENS,
            "severe_repetition_threshold": 0.8,
            "minimum_tokens_for_repetition": 16,
        },
        "selection_metadata": selection_metadata,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "token_length_p50_p90_p95_p99_max": (
            [percentile(token_counts, p) for p in (50, 90, 95, 99)] + [max(token_counts)]
            if token_counts
            else [None] * 5
        ),
        "unknown_token_count": sum(
            item["flags"].get("unknown_token_count", 0) for item in per_sample
        ),
        "per_sample": per_sample,
    }
    (output_dir / "pseudo_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "pseudo_train_filtered.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in filtered),
        encoding="utf-8",
    )
    return filtered, report


def establish_common_pseudo_intersection(
    shared_root: Path,
    selected_rows: list[dict],
    teacher_a_filtered_path: Path,
    teacher_b_filtered_path: Path,
) -> tuple[set[str] | None, dict | None]:
    """Create one immutable usable-ID mask shared by both teacher students.

    The first teacher pass intentionally stops until both filtered manifests exist.
    This prevents Student A and Student B from silently training on different
    subsets when one teacher has more invalid outputs than the other.
    """

    if not teacher_a_filtered_path.is_file() or not teacher_b_filtered_path.is_file():
        return None, None
    selected_ids = {str(row["sample_id"]) for row in selected_rows}
    teacher_a = load_jsonl(teacher_a_filtered_path)
    teacher_b = load_jsonl(teacher_b_filtered_path)
    common_ids = sorted(set(teacher_a) & set(teacher_b) & selected_ids)
    if not common_ids:
        raise RuntimeError("Teacher A and Teacher B have no common usable pseudo-training IDs")
    for sample_id in common_ids:
        if teacher_a[sample_id].get("audio_sha256") != teacher_b[sample_id].get("audio_sha256"):
            raise RuntimeError(f"audio identity mismatch between teachers for {sample_id}")
    metadata = {
        "schema_version": 1,
        "selected_source_count": len(selected_ids),
        "teacher_a_usable_count": len(teacher_a),
        "teacher_b_usable_count": len(teacher_b),
        "common_usable_count": len(common_ids),
        "sample_ids_sha256": sha256_text("\n".join(common_ids) + "\n"),
        "sample_ids": common_ids,
        "teacher_a_filtered_sha256": sha256_file(teacher_a_filtered_path),
        "teacher_b_filtered_sha256": sha256_file(teacher_b_filtered_path),
    }
    path = shared_root / "common_pseudo_selection.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(
                "common_pseudo_selection.json already exists but does not match the current "
                "Teacher A/Teacher B filtered manifests"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return set(common_ids), metadata


def ensure_selected_manifest(selection_root: Path, expected_count: int, require_existing: bool) -> tuple[list[dict], dict]:
    """Validate the shared selection without rereading 10k Drive audio files by default.

    Google Drive FUSE makes a full byte-for-byte hash pass extremely slow.  The
    immutable manifest and per-file existence checks are the normal Colab path;
    set VERIFY_SHARED_AUDIO_HASHES=1 when an explicit full integrity audit is
    needed.
    """
    manifest_path = selection_root / "selected_manifest.jsonl"
    metadata_path = selection_root / "selection_metadata.json"
    if require_existing and not manifest_path.is_file():
        raise FileNotFoundError(
            f"shared selection is missing: {manifest_path}. Run Student A first."
        )
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(rows) != expected_count or metadata.get("selected_count") != expected_count:
        raise RuntimeError("shared pseudo selection does not contain the requested number of rows")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("shared pseudo selection contains duplicate sample IDs")
    if metadata.get("sample_ids_sha256") != sha256_text("\n".join(sorted(ids)) + "\n"):
        raise RuntimeError("shared pseudo selection sample ID hash is inconsistent")
    ids_file = selection_root / "selected_train_ids.txt"
    if not ids_file.is_file() or ids_file.read_text(encoding="utf-8") != "\n".join(ids) + "\n":
        raise RuntimeError("shared pseudo selection ID list is inconsistent")
    if metadata.get("selection_seed") != SELECTION_SEED:
        raise RuntimeError("shared pseudo selection seed differs")
    verify_audio_hashes = os.environ.get("VERIFY_SHARED_AUDIO_HASHES", "0") == "1"
    for row in rows:
        audio_path = selection_root / row["audio_path"]
        if not audio_path.is_file():
            raise RuntimeError(f"shared pseudo audio is missing or changed: {audio_path}")
        if verify_audio_hashes and sha256_file(audio_path) != row["audio_sha256"]:
            raise RuntimeError(f"shared pseudo audio hash differs: {audio_path}")
    print("shared_audio_hash_verification=", "full" if verify_audio_hashes else "manifest_only")
    return rows, metadata


def select_and_materialize_wordproject(
    source_root: Path,
    selection_root: Path,
    token: str,
    requested: int,
) -> tuple[list[dict], dict]:
    """Scan WordProject, choose a stable train-only subset, then materialize its WAV files."""

    from seamtranslation.preprocessing.audio import decode_compressed_audio, encode_wav, resample
    from seamtranslation.preprocessing.samanantar import normalize_text, rejection_reason

    held_out = benchmark_ids(source_root)
    guards = existing_real_guards(source_root)
    candidates: list[dict] = []
    rejected: Counter[str] = Counter()
    seen_original: set[str] = set()
    scanned = 0
    print("dataset_scan=starting; this scans WordProject metadata and audio once")
    for source_index, row in enumerate(stream_wordproject(token)):
        scanned += 1
        try:
            raw_audio = _audio_bytes(row["chunked_audio_filepath"])
            duration = float(row["duration"])
            alignment = float(row["alignment_score"])
            translation = float(row["hi_mining_score"])
            recording_id = str(row.get("audio_filepath") or "")
            source_text = normalize_text(str(row.get("text") or ""))
            source_asr_text = normalize_text(str(row.get("pred_text") or ""))
            human_hindi = normalize_text(str(row.get("hi_text") or ""))
            original_hash = sha256_bytes(raw_audio)
            if not 1.0 <= duration <= 20.0:
                rejected["duration"] += 1
                continue
            if alignment < 0.9:
                rejected["alignment_score"] += 1
                continue
            if translation < 0.8:
                rejected["translation_score"] += 1
                continue
            if reason := rejection_reason(source_text, human_hindi):
                rejected[reason] += 1
                continue
            if _recording_split(recording_id) != "train":
                rejected["recording_validation_split"] += 1
                continue
            if original_hash in seen_original:
                rejected["duplicate_audio"] += 1
                continue
            seen_original.add(original_hash)
            recording_hash = sha256_text(recording_id)
            if (
                original_hash in guards["original_audio_sha256"]
                or recording_hash in guards["recording_hash"]
            ):
                rejected["overlap_with_existing_real_data"] += 1
                continue
            source_key = f"{DATASET_CONFIG}:{source_index}:{recording_id}:{original_hash}"
            identity = sha256_text(f"{DATASET_REVISION}:{source_key}")
            sample_id = f"wordproject-en-hi-pseudo-{identity[:24]}"
            if sample_id in guards["ids"] or sample_id in held_out:
                rejected["held_out_or_existing_id"] += 1
                continue
            candidates.append(
                {
                    "sample_id": sample_id,
                    "source_key": source_key,
                    "source_row": source_index,
                    "source_shard": DATASET_CONFIG,
                    "recording_hash": recording_hash,
                    "audio_origin": "real",
                    "source_dataset": DATASET_ID,
                    "source_dataset_revision": DATASET_REVISION,
                    "source_license": DATASET_LICENSE,
                    "source_language": "en",
                    "target_language": "hi",
                    "source_text": source_text,
                    "source_asr_text": source_asr_text,
                    "human_reference_hindi": human_hindi,
                    "duration_seconds": duration,
                    "sample_rate": None,
                    "alignment_score": alignment,
                    "translation_score": translation,
                    "original_audio_sha256": original_hash,
                    "content_sha256": None,
                    "audio_sha256": None,
                    "selection_key": sha256_text(f"{SELECTION_SEED}:{source_key}"),
                }
            )
        except Exception as exc:
            rejected[type(exc).__name__] += 1
    if len(candidates) < requested:
        raise RuntimeError(
            json.dumps(
                {
                    "reason": "not_enough_clean_wordproject_examples",
                    "scanned": scanned,
                    "accepted_candidates": len(candidates),
                    "requested": requested,
                    "rejected": dict(sorted(rejected.items())),
                },
                indent=2,
            )
        )
    selected = sorted(candidates, key=lambda row: row["selection_key"])[:requested]
    selected.sort(key=lambda row: row["sample_id"])
    selection_root.mkdir(parents=True, exist_ok=True)
    audio_root = selection_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []
    selected_keys = {candidate["source_key"]: candidate for candidate in selected}
    print("materializing_selected_audio=", len(selected_keys))
    for source_index, row in enumerate(stream_wordproject(token)):
        raw_audio = _audio_bytes(row["chunked_audio_filepath"])
        recording_id = str(row.get("audio_filepath") or "")
        original_hash = sha256_bytes(raw_audio)
        source_key = f"{DATASET_CONFIG}:{source_index}:{recording_id}:{original_hash}"
        candidate = selected_keys.get(source_key)
        if candidate is None:
            continue
        from seamtranslation.preprocessing.audio import decode_compressed_audio, encode_wav, resample

        converted = resample(decode_compressed_audio(raw_audio), target_rate=TARGET_RATE)
        wav_bytes = encode_wav(converted)
        content_hash = sha256_bytes(wav_bytes)
        if content_hash in guards["content_sha256"]:
            raise RuntimeError(
                f"selected source audio becomes a duplicate of the real dataset after conversion: "
                f"{candidate['sample_id']}"
            )
        relative = Path("audio") / f"{candidate['sample_id']}.wav"
        path = selection_root / relative
        if not path.exists():
            path.write_bytes(wav_bytes)
        candidate = dict(candidate)
        candidate.update(
            {
                "duration_seconds": len(converted.waveform) / TARGET_RATE,
                "sample_rate": TARGET_RATE,
                "content_sha256": content_hash,
                "audio_sha256": content_hash,
            }
        )
        output_rows.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in {"selection_key"}
            }
            | {"audio_path": relative.as_posix(), "split": "train"},
        )
        if len(output_rows) == requested:
            break
    if len(output_rows) != requested:
        raise RuntimeError("the second WordProject pass did not materialize every selected sample")
    manifest_path = selection_root / "selected_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    (selection_root / "selected_train_ids.txt").write_text(
        "\n".join(row["sample_id"] for row in output_rows) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "selection_seed": SELECTION_SEED,
        "requested_count": requested,
        "selected_count": len(output_rows),
        "scanned": scanned,
        "candidate_count": len(candidates),
        "rejected": dict(sorted(rejected.items())),
        "source_shard_identities": sorted({row["source_shard"] for row in output_rows}),
        "source_shard_identity_note": (
            "WordProject streaming rows do not expose parquet filenames; source_shard is the "
            "pinned dataset configuration fallback."
        ),
        "filters": {
            "duration_seconds": [1.0, 20.0],
            "minimum_alignment_score": 0.9,
            "minimum_translation_score": 0.8,
            "recording_split": "train",
            "excluded_existing_real_manifest_sha256": EXPECTED_REAL_MANIFEST_SHA256,
            "excluded_teacher_benchmark_sample_ids_sha256": BENCHMARK_SAMPLE_IDS_SHA256,
        },
        "sample_ids_sha256": sha256_text(
            "\n".join(sorted(row["sample_id"] for row in output_rows)) + "\n"
        ),
        "manifest_sha256": sha256_file(manifest_path),
    }
    (selection_root / "selection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_rows, metadata


def materialize_real_dataset(source_root: Path, shared_root: Path, token: str) -> Path:
    """Recreate the exact 2K real manifest's audio in Drive without uploading 526 MB."""

    from seamtranslation.preprocessing.audio import decode_compressed_audio, encode_wav, resample

    output_manifest = shared_root / "real_data/manifest.jsonl"
    metadata_path = shared_root / "real_data/metadata.json"
    if output_manifest.is_file() and metadata_path.is_file():
        return output_manifest
    source_manifest = source_root / REAL_MANIFEST_RELATIVE
    source_rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line]
    needed = {str(row["original_audio_sha256"]): row for row in source_rows}
    real_audio_root = shared_root / "real_data/audio"
    real_audio_root.mkdir(parents=True, exist_ok=True)
    for row_index, row in enumerate(stream_wordproject(token)):
        raw_audio = _audio_bytes(row["chunked_audio_filepath"])
        original_hash = sha256_bytes(raw_audio)
        target = needed.get(original_hash)
        if target is None:
            continue
        converted = resample(decode_compressed_audio(raw_audio), target_rate=TARGET_RATE)
        wav = encode_wav(converted)
        path = real_audio_root / f"{target['id']}.wav"
        if not path.exists():
            path.write_bytes(wav)
        needed.pop(original_hash)
        if not needed:
            break
        if row_index and row_index % 500 == 0:
            print("real_audio_materialized=", len(source_rows) - len(needed), "/", len(source_rows))
    if needed:
        raise RuntimeError(f"could not rematerialize {len(needed)} real WordProject audio files")
    output_rows = []
    for row in source_rows:
        output_rows.append(
            dict(row)
            | {
                "audio_path": f"audio/{row['id']}.wav",
                "label_origin": "real",
                "human_target_text": row["target_text"],
            }
        )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_manifest_sha256": EXPECTED_REAL_MANIFEST_SHA256,
                "manifest_sha256": sha256_file(output_manifest),
                "examples": len(output_rows),
                "real_train_examples": sum(row["split"] == "train" for row in output_rows),
                "real_validation_examples": sum(row["split"] == "validation" for row in output_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_manifest


def write_training_manifests(
    shared_root: Path,
    output_root: Path,
    pseudo_rows: list[dict],
    real_manifest: Path,
    teacher_tag: str,
) -> tuple[Path, Path]:
    real_rows = [json.loads(line) for line in real_manifest.read_text(encoding="utf-8").splitlines() if line]
    pseudo_path = shared_root / "training_manifests" / f"{teacher_tag}_pseudo_pretrain.jsonl"
    real_path = shared_root / "training_manifests" / f"{teacher_tag}_real_finetune.jsonl"
    pseudo_rows_for_training = []
    for row in pseudo_rows:
        absolute_audio = shared_root / row["audio_path"]
        pseudo_rows_for_training.append(
            dict(row)
            | {
                "audio_path": os.path.relpath(absolute_audio, pseudo_path.parent),
                "target_text": row["teacher_hindi_normalized"],
                "human_target_text": row["human_reference_hindi"],
                "label_origin": "pseudo",
                "split": "train",
            }
        )
    validation_rows = [row for row in real_rows if row["split"] == "validation"]
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    validation_rows = [
        dict(row)
        | {
            "audio_path": os.path.relpath(real_manifest.parent / row["audio_path"], pseudo_path.parent)
        }
        for row in validation_rows
    ]
    pseudo_manifest_rows = pseudo_rows_for_training + validation_rows
    pseudo_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pseudo_manifest_rows),
        encoding="utf-8",
    )
    real_rows_for_training = [
        dict(row)
        | {"audio_path": os.path.relpath(real_manifest.parent / row["audio_path"], real_path.parent)}
        for row in real_rows
    ]
    real_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in real_rows_for_training),
        encoding="utf-8",
    )
    return pseudo_path, real_path


def run_two_stage_student(
    source_root: Path,
    shared_root: Path,
    output_root: Path,
    pseudo_manifest: Path,
    real_manifest: Path,
    pseudo_count: int,
    teacher_metadata: dict,
    pseudo_steps: int = 2_000,
    real_steps: int = 1_000,
) -> dict:
    import torch

    from seamtranslation.tokenization import HindiTokenizer
    from train.train_semantic import train

    tokenizer_path = source_root / TOKENIZER_RELATIVE
    if sha256_file(tokenizer_path) != TOKENIZER_SHA256:
        raise RuntimeError("Hindi-2K tokenizer hash differs from the baseline tokenizer")
    base_config = json.loads((source_root / MODEL_CONFIG_RELATIVE).read_text(encoding="utf-8"))
    tokenizer = HindiTokenizer(tokenizer_path)
    output_root.mkdir(parents=True, exist_ok=True)
    stage1_root = output_root / "training/pseudo_pretrain"
    stage2_root = output_root / "training/real_finetune"
    stage1_config = json.loads(json.dumps(base_config))
    stage1_config["training"].update(
        {
            "max_examples": pseudo_count,
            "max_validation_examples": 264,
            "max_steps": pseudo_steps,
            "seed": 17,
            "evaluation_split": "validation",
        }
    )
    stage2_config = json.loads(json.dumps(base_config))
    stage2_config["training"].update(
        {
            "max_examples": 1736,
            "max_validation_examples": 264,
            "max_steps": pseudo_steps + real_steps,
            "seed": 17,
            "evaluation_split": "validation",
        }
    )
    stage1_config_path = output_root / "training/pseudo_pretrain_config.json"
    stage2_config_path = output_root / "training/real_finetune_config.json"
    stage1_config_path.parent.mkdir(parents=True, exist_ok=True)
    stage1_config_path.write_text(json.dumps(stage1_config, indent=2, sort_keys=True) + "\n")
    stage2_config_path.write_text(json.dumps(stage2_config, indent=2, sort_keys=True) + "\n")

    stage1_checkpoint = stage1_root / "checkpoint.pt"
    if not (stage1_root / "metadata.json").is_file():
        train(
            stage1_config_path,
            pseudo_manifest,
            tokenizer_path,
            stage1_root,
            resume_path=stage1_checkpoint if stage1_checkpoint.is_file() else None,
        )
    else:
        print("stage1=complete; reusing existing metadata")
    if not stage1_checkpoint.is_file():
        raise RuntimeError("pseudo-pretraining did not produce checkpoint.pt")

    stage2_checkpoint = stage2_root / "checkpoint.pt"
    if not (stage2_root / "metadata.json").is_file():
        train(
            stage2_config_path,
            real_manifest,
            tokenizer_path,
            stage2_root,
            resume_path=stage2_checkpoint if stage2_checkpoint.is_file() else stage1_checkpoint,
            allow_manifest_change_on_resume=not stage2_checkpoint.is_file(),
            reset_best_validation=not stage2_checkpoint.is_file(),
        )
    else:
        print("stage2=complete; reusing existing metadata")
    if not stage2_checkpoint.is_file():
        raise RuntimeError("real fine-tuning did not produce checkpoint.pt")
    shutil.copyfile(stage2_checkpoint, stage2_root / "last.pt")
    if (stage2_root / "best.pt").is_file():
        shutil.copyfile(stage2_root / "best.pt", output_root / "training/best.pt")
    shutil.copyfile(stage2_checkpoint, output_root / "training/last.pt")
    metadata = json.loads((stage2_root / "metadata.json").read_text(encoding="utf-8"))
    run_metadata = {
        "schema_version": 1,
        "teacher_metadata": teacher_metadata,
        "student_model_parameters": metadata.get("parameters_trained"),
        "student_seed": 17,
        "student_config": base_config,
        "pseudo_pretrain_steps": pseudo_steps,
        "real_finetune_steps": real_steps,
        "pseudo_manifest_sha256": sha256_file(pseudo_manifest),
        "real_manifest_sha256": sha256_file(real_manifest),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "device": metadata.get("device"),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
        "cuda_peak_memory_gib": (
            float(torch.cuda.max_memory_allocated() / 1024**3)
            if torch.cuda.is_available()
            else None
        ),
        "checkpoint_sha256": sha256_file(stage2_checkpoint),
    }
    (output_root / "training/run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "training/metrics.json").write_text(
        json.dumps(
            {
                "validation_history": metadata.get("validation_history", []),
                "generation_history": metadata.get("generation_history", []),
                "best_validation_loss": metadata.get("best_validation_loss"),
                "parameters": metadata.get("parameters_trained"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_metadata


def archive_experiment(root: Path) -> Path:
    archive_base = root.parent / root.name
    archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=root.parent, base_dir=root.name))
    print("artifact_archive=", archive)
    return archive
