"""Semantic-only model used before speech-unit and vocoder training."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from seamtranslation.models.config import ModelConfig
from seamtranslation.models.conformer import SpeechEncoder
from seamtranslation.models.frontend import ConvSubsampler, LogMelFrontend
from seamtranslation.models.semantic import SemanticDecoder


class SpeechSemanticModel(nn.Module):
    """English waveform to Hindi semantic-token logits, without unused S2S modules."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.frontend = LogMelFrontend(config)
        self.subsampler = ConvSubsampler(config.feature_bins, config.model_dim)
        self.encoder = SpeechEncoder(
            dim=config.model_dim,
            layers=config.encoder_layers,
            heads=config.encoder_heads,
            ffn_dim=config.encoder_ffn_dim,
            kernel_size=config.conformer_kernel_size,
            dropout=config.dropout,
        )
        self.semantic_decoder = SemanticDecoder(
            vocab_size=config.semantic_vocab_size,
            dim=config.model_dim,
            layers=config.semantic_layers,
            heads=config.semantic_heads,
            ffn_dim=config.semantic_ffn_dim,
            max_tokens=config.max_semantic_tokens,
            dropout=config.dropout,
            pad_token_id=config.pad_token_id,
        )

    def forward(
        self, waveform: Tensor, waveform_lengths: Tensor, semantic_input_ids: Tensor
    ) -> Tensor:
        features, feature_lengths = self.frontend(waveform, waveform_lengths)
        hidden, encoded_lengths = self.subsampler(features, feature_lengths)
        memory, memory_padding = self.encoder(hidden, encoded_lengths)
        return self.semantic_decoder(semantic_input_ids, memory, memory_padding)

    @torch.inference_mode()
    def generate(self, waveform: Tensor, waveform_lengths: Tensor) -> Tensor:
        features, feature_lengths = self.frontend(waveform, waveform_lengths)
        hidden, encoded_lengths = self.subsampler(features, feature_lengths)
        memory, memory_padding = self.encoder(hidden, encoded_lengths)
        return self.semantic_decoder.generate(
            memory,
            memory_padding,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
