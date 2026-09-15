from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

@dataclass(frozen=True)
class UnitDecoderOutput:
    logits: Tensor
    durations: Tensor
    lengths: Tensor


class UnitDecoder(nn.Module):
    def __init__(self, *, semantic_vocab_size: int, unit_vocab_size: int, dim: int,
                 layers: int, heads: int, ffn_dim: int, dropout: float,
                 pad_token_id: int) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(semantic_vocab_size, dim, padding_idx=pad_token_id)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.semantic_encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim))
        self.duration_predictor = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )
        self.unit_projection = nn.Linear(dim, unit_vocab_size)

    @staticmethod
    def _length_regulate(hidden: Tensor, durations: Tensor) -> tuple[Tensor, Tensor]:
        sequences: list[Tensor] = []
        lengths: list[int] = []
        for sample, sample_durations in zip(hidden, durations, strict=True):
            expanded = torch.repeat_interleave(sample, sample_durations, dim=0)
            if expanded.shape[0] == 0:
                expanded = sample[:1]
            sequences.append(expanded)
            lengths.append(expanded.shape[0])
        max_length = max(lengths)
        padded = hidden.new_zeros((hidden.shape[0], max_length, hidden.shape[-1]))
        for index, sequence in enumerate(sequences):
            padded[index, : sequence.shape[0]] = sequence
        return padded, torch.tensor(lengths, device=hidden.device, dtype=torch.long)

    def forward(self, semantic_tokens: Tensor, durations: Tensor | None = None
                ) -> UnitDecoderOutput:
        semantic_padding = semantic_tokens.eq(self.pad_token_id)
        hidden = self.semantic_encoder(
            self.embedding(semantic_tokens), src_key_padding_mask=semantic_padding
        )
        predicted = nn.functional.softplus(self.duration_predictor(hidden).squeeze(-1))
        predicted = predicted.masked_fill(semantic_padding, 0.0)
        if durations is None:
            discrete_durations = predicted.round().clamp(min=1).long()
            discrete_durations = discrete_durations.masked_fill(semantic_padding, 0)
        else:
            if durations.shape != semantic_tokens.shape:
                raise ValueError("durations must match semantic_tokens shape")
            discrete_durations = durations.masked_fill(semantic_padding, 0).long()
        expanded, unit_lengths = self._length_regulate(hidden, discrete_durations)
        return UnitDecoderOutput(
            logits=self.unit_projection(expanded),
            durations=predicted,
            lengths=unit_lengths,
        )
