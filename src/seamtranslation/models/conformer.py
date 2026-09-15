from __future__ import annotations

import torch
from torch import Tensor, nn


def lengths_to_padding_mask(lengths: Tensor, max_length: int) -> Tensor:
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) >= lengths.unsqueeze(1)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.layer_norm(inputs).transpose(1, 2)
        hidden = nn.functional.glu(self.pointwise_in(hidden), dim=1)
        hidden = nn.functional.silu(self.batch_norm(self.depthwise(hidden)))
        return self.dropout(self.pointwise_out(hidden).transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, kernel_size: int,
                 dropout: float) -> None:
        super().__init__()
        self.ffn_in = FeedForwardModule(dim, ffn_dim, dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution = ConvolutionModule(dim, kernel_size, dropout)
        self.ffn_out = FeedForwardModule(dim, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor | None
                ) -> Tensor:
        hidden = inputs + 0.5 * self.ffn_in(inputs)
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        hidden = hidden + self.attention_dropout(attended)
        hidden = hidden + self.convolution(hidden)
        hidden = hidden + 0.5 * self.ffn_out(hidden)
        return self.final_norm(hidden).masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SpeechEncoder(nn.Module):
    def __init__(self, *, dim: int, layers: int, heads: int, ffn_dim: int,
                 kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ConformerBlock(dim, heads, ffn_dim, kernel_size, dropout) for _ in range(layers)
        )

    @staticmethod
    def _chunk_mask(steps: int, device: torch.device, left_context: int | None,
                    right_context: int | None) -> Tensor | None:
        if left_context is None and right_context is None:
            return None
        positions = torch.arange(steps, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        allowed = torch.ones((steps, steps), dtype=torch.bool, device=device)
        if left_context is not None:
            allowed &= distance >= -left_context
        if right_context is not None:
            allowed &= distance <= right_context
        return ~allowed

    def forward(self, inputs: Tensor, lengths: Tensor, *, left_context: int | None = None,
                right_context: int | None = None) -> tuple[Tensor, Tensor]:
        padding_mask = lengths_to_padding_mask(lengths, inputs.shape[1])
        attention_mask = self._chunk_mask(
            inputs.shape[1], inputs.device, left_context, right_context
        )
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden, padding_mask, attention_mask)
        return hidden, padding_mask

    def reset_state(self) -> None:
        """Reserved for the bounded streaming cache; offline encoder is stateless."""
