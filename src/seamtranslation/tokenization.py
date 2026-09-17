"""Hindi SentencePiece tokenizer with model-compatible special-token IDs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TokenizerSpecialIds:
    pad: int = 0
    bos: int = 1
    eos: int = 2
    unknown: int = 3


class HindiTokenizer:
    def __init__(self, model_path: Path | str) -> None:
        try:
            import sentencepiece as sentencepiece
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "tokenization requires: python -m pip install -e '.[training]'"
            ) from exc
        self._processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        self.special_ids = TokenizerSpecialIds(
            pad=self._processor.pad_id(),
            bos=self._processor.bos_id(),
            eos=self._processor.eos_id(),
            unknown=self._processor.unk_id(),
        )
        if self.special_ids != TokenizerSpecialIds():
            raise ValueError(f"incompatible tokenizer special IDs: {self.special_ids}")

    @property
    def vocabulary_size(self) -> int:
        return self._processor.vocab_size()

    def encode(self, text: str) -> list[int]:
        return list(self._processor.encode(text, out_type=int))

    def decode(self, token_ids: Iterable[int]) -> str:
        return self._processor.decode(list(token_ids))
