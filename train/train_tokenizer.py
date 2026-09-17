"""Train the versioned Hindi semantic tokenizer from JSONL manifests."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import sentencepiece as spm

from seamtranslation.tokenization import HindiTokenizer


def train_tokenizer(
    manifests: list[Path], output: Path, *, vocabulary_size: int, text_field: str
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if vocabulary_size < 32:
        raise ValueError("vocabulary_size must be at least 32")
    output.mkdir(parents=True)
    manifest_hashes = {str(path): _sha256_file(path) for path in manifests}
    sentence_count = 0
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as corpus:
        corpus_path = Path(corpus.name)
        for manifest in manifests:
            with manifest.open(encoding="utf-8") as stream:
                for line in stream:
                    text = json.loads(line).get(text_field)
                    if isinstance(text, str) and text.strip():
                        corpus.write(text.strip() + "\n")
                        sentence_count += 1
    if sentence_count == 0:
        corpus_path.unlink(missing_ok=True)
        raise ValueError(f"no non-empty {text_field!r} values found")

    prefix = output / "hindi"
    try:
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(prefix),
            vocab_size=vocabulary_size,
            model_type="unigram",
            character_coverage=1.0,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            hard_vocab_limit=False,
            shuffle_input_sentence=False,
            num_threads=1,
        )
    finally:
        corpus_path.unlink(missing_ok=True)

    tokenizer = HindiTokenizer(prefix.with_suffix(".model"))
    metadata = {
        "sentence_count": sentence_count,
        "requested_vocabulary_size": vocabulary_size,
        "actual_vocabulary_size": tokenizer.vocabulary_size,
        "text_field": text_field,
        "manifest_sha256": manifest_hashes,
        "model_sha256": _sha256_file(prefix.with_suffix(".model")),
        "special_ids": {
            "pad": tokenizer.special_ids.pad,
            "bos": tokenizer.special_ids.bos,
            "eos": tokenizer.special_ids.eos,
            "unknown": tokenizer.special_ids.unknown,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary-size", type=int, default=8_000)
    parser.add_argument("--text-field", default="target_text")
    args = parser.parse_args()
    metadata = train_tokenizer(
        args.manifest,
        args.output,
        vocabulary_size=args.vocabulary_size,
        text_field=args.text_field,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
