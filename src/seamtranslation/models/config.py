from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    sample_rate: int = 16_000
    feature_bins: int = 80
    n_fft: int = 400
    hop_length: int = 160
    model_dim: int = 384
    encoder_layers: int = 12
    encoder_heads: int = 6
    encoder_ffn_dim: int = 1_536
    conformer_kernel_size: int = 15
    semantic_layers: int = 6
    semantic_heads: int = 6
    semantic_ffn_dim: int = 1_536
    semantic_vocab_size: int = 8_000
    unit_layers: int = 4
    unit_heads: int = 4
    unit_ffn_dim: int = 1_024
    unit_dim: int = 256
    unit_vocab_size: int = 1_000
    vocoder_channels: int = 256
    max_semantic_tokens: int = 128
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        positive = (
            "sample_rate",
            "feature_bins",
            "n_fft",
            "hop_length",
            "model_dim",
            "encoder_layers",
            "encoder_heads",
            "encoder_ffn_dim",
            "semantic_layers",
            "semantic_heads",
            "semantic_ffn_dim",
            "semantic_vocab_size",
            "unit_layers",
            "unit_heads",
            "unit_ffn_dim",
            "unit_dim",
            "unit_vocab_size",
            "vocoder_channels",
            "max_semantic_tokens",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.encoder_heads:
            raise ValueError("model_dim must be divisible by encoder_heads")
        if self.model_dim % self.semantic_heads:
            raise ValueError("model_dim must be divisible by semantic_heads")
        if self.unit_dim % self.unit_heads:
            raise ValueError("unit_dim must be divisible by unit_heads")
        if self.conformer_kernel_size % 2 == 0:
            raise ValueError("conformer_kernel_size must be odd")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        if len(special_ids) != 3 or min(special_ids) < 0:
            raise ValueError("pad, bos and eos IDs must be distinct and non-negative")
        if max(special_ids) >= self.semantic_vocab_size:
            raise ValueError("special token IDs must fit semantic_vocab_size")
