"""Data and loss functions for English-speech-to-Hindi-semantic training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from seamtranslation.models import SpeechSemanticModel
from seamtranslation.preprocessing import decode_wav
from seamtranslation.tokenization import HindiTokenizer


@dataclass(frozen=True, slots=True)
class SemanticExample:
    waveform: Tensor
    target_text: str


@dataclass(frozen=True, slots=True)
class SemanticBatch:
    waveforms: Tensor
    waveform_lengths: Tensor
    decoder_input_ids: Tensor
    labels: Tensor

    def to(self, device: torch.device) -> SemanticBatch:
        return SemanticBatch(
            waveforms=self.waveforms.to(device),
            waveform_lengths=self.waveform_lengths.to(device),
            decoder_input_ids=self.decoder_input_ids.to(device),
            labels=self.labels.to(device),
        )


class WordProjectSemanticDataset(Dataset[SemanticExample]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        max_examples: int | None = None,
    ) -> None:
        self.root = manifest_path.parent
        self.rows: list[dict[str, object]] = []
        with manifest_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["split"] == split:
                    self.rows.append(row)
                    if max_examples is not None and len(self.rows) >= max_examples:
                        break
        if not self.rows:
            raise ValueError(f"manifest contains no {split!r} examples")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> SemanticExample:
        row = self.rows[index]
        audio = decode_wav((self.root / str(row["audio_path"])).read_bytes())
        if audio.sample_rate != 16_000:
            raise ValueError(f"expected 16 kHz training audio, got {audio.sample_rate}")
        return SemanticExample(
            waveform=torch.from_numpy(audio.waveform.copy()),
            target_text=str(row["target_text"]),
        )


def collate_semantic_batch(
    examples: list[SemanticExample],
    tokenizer: HindiTokenizer,
    *,
    max_tokens: int,
) -> SemanticBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    pad = tokenizer.special_ids.pad
    bos = tokenizer.special_ids.bos
    eos = tokenizer.special_ids.eos
    waveform_lengths = torch.tensor([len(row.waveform) for row in examples])
    waveforms = torch.zeros(len(examples), int(waveform_lengths.max()), dtype=torch.float32)
    decoder_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for index, example in enumerate(examples):
        waveforms[index, : len(example.waveform)] = example.waveform
        pieces = tokenizer.encode(example.target_text)[: max_tokens - 1]
        decoder_rows.append([bos, *pieces])
        label_rows.append([*pieces, eos])

    token_length = max(len(row) for row in decoder_rows)
    decoder_input_ids = torch.full((len(examples), token_length), pad, dtype=torch.long)
    labels = torch.full((len(examples), token_length), pad, dtype=torch.long)
    for index, (decoder_row, label_row) in enumerate(zip(decoder_rows, label_rows, strict=True)):
        decoder_input_ids[index, : len(decoder_row)] = torch.tensor(decoder_row)
        labels[index, : len(label_row)] = torch.tensor(label_row)
    return SemanticBatch(waveforms, waveform_lengths, decoder_input_ids, labels)


def semantic_loss(model: SpeechSemanticModel, batch: SemanticBatch) -> Tensor:
    logits = model(
        batch.waveforms,
        batch.waveform_lengths,
        batch.decoder_input_ids,
    )
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.labels.reshape(-1),
        ignore_index=model.config.pad_token_id,
    )
