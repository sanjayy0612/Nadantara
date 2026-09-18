"""Create a deterministic, portable teacher-benchmark bundle.

The source manifest is never modified. The bundle contains only the selected
audio bytes and a portable JSONL manifest with human references reserved for
evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import tarfile


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260918


@dataclass(frozen=True, slots=True)
class BundleSummary:
    schema_version: int
    source_manifest: str
    source_manifest_sha256: str
    source_dataset: str
    source_revision: str
    split: str
    seed: int
    requested_examples: int
    examples: int
    sample_ids_sha256: str
    manifest_sha256: str
    git_commit: str | None
    archive: str | None


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(manifest: Path, split: str, limit: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            if not row.get("id") or not row.get("audio_path"):
                raise ValueError(f"row {line_number} lacks id or audio_path")
            rows.append(row)
    if len(rows) < limit:
        raise ValueError(f"only {len(rows)} {split!r} rows available; need {limit}")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("source manifest contains duplicate IDs in the selected split")
    rng = random.Random(seed)
    return sorted(rng.sample(rows, limit), key=lambda row: str(row["id"]))


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_bundle(
    manifest: Path,
    output: Path,
    *,
    split: str,
    limit: int,
    seed: int,
    archive: Path | None,
) -> BundleSummary:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if limit <= 0:
        raise ValueError("limit must be positive")

    rows = selected_rows(manifest, split, limit, seed)
    source_root = manifest.parent
    source_manifest_sha256 = sha256_file(manifest)
    output.mkdir(parents=True)
    audio_root = output / "audio"
    audio_root.mkdir()
    portable_rows: list[dict[str, object]] = []

    for row in rows:
        sample_id = str(row["id"])
        source_audio = (source_root / str(row["audio_path"])).resolve()
        if not source_audio.is_file():
            raise FileNotFoundError(source_audio)
        if source_root.resolve() not in source_audio.parents:
            raise ValueError(f"audio path escapes manifest root: {row['audio_path']}")
        audio_bytes = source_audio.read_bytes()
        audio_name = f"{sample_id}.wav"
        (audio_root / audio_name).write_bytes(audio_bytes)
        portable_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "audio_path": f"audio/{audio_name}",
                "audio_sha256": sha256(audio_bytes).hexdigest(),
                "source_manifest_sha256": source_manifest_sha256,
                "source_dataset": row.get("dataset_id"),
                "source_dataset_revision": row.get("dataset_revision"),
                "source_split": row.get("split"),
                "source_subset": row.get("source_subset"),
                "source_license": row.get("source_license"),
                "source_url": row.get("source_url"),
                "duration_seconds": row.get("duration_seconds"),
                "sample_rate": row.get("sample_rate"),
                "reference_hindi": row.get("target_text"),
                "source_text": row.get("source_text"),
                "source_asr_text": row.get("source_asr_text"),
                "audio_origin": row.get("audio_origin"),
                "translation_origin": row.get("translation_origin"),
            }
        )

    manifest_out = output / "benchmark_manifest.jsonl"
    with manifest_out.open("w", encoding="utf-8") as stream:
        for row in portable_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    ids_text = "\n".join(str(row["sample_id"]) for row in portable_rows) + "\n"
    (output / "sample_ids.txt").write_text(ids_text, encoding="utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "source_dataset": portable_rows[0]["source_dataset"],
        "source_revision": portable_rows[0]["source_dataset_revision"],
        "split": split,
        "seed": seed,
        "requested_examples": limit,
        "examples": len(portable_rows),
        "sample_ids_sha256": sha256(ids_text.encode()).hexdigest(),
        "manifest_sha256": sha256_file(manifest_out),
        "git_commit": git_commit(),
        "archive": str(archive) if archive else None,
    }
    (output / "bundle_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if archive:
        if archive.exists():
            raise FileExistsError(f"archive already exists: {archive}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(output, arcname=output.name)

    return BundleSummary(**metadata)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = build_bundle(
        args.manifest,
        args.output,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        archive=args.archive,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
