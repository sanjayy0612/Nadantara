import json
from pathlib import Path

from seamtranslation.tokenization import HindiTokenizer, TokenizerSpecialIds
from train.train_tokenizer import train_tokenizer


def test_train_and_load_hindi_tokenizer(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"target_text": "यह एक परीक्षण वाक्य है"},
        {"target_text": "हिन्दी भाषा सुंदर है"},
        {"target_text": "यह दूसरा परीक्षण वाक्य है"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "tokenizer"

    metadata = train_tokenizer([manifest], output, vocabulary_size=64, text_field="target_text")
    tokenizer = HindiTokenizer(output / "hindi.model")
    encoded = tokenizer.encode("हिन्दी भाषा सुंदर है")

    assert metadata["sentence_count"] == 3
    assert tokenizer.special_ids == TokenizerSpecialIds()
    assert encoded
    assert tokenizer.decode(encoded) == "हिन्दी भाषा सुंदर है"
