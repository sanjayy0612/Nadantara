from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .conformer import SpeechEncoder
from .frontend import ConvSubsampler, LogMelFrontend
from .semantic import SemanticDecoder
from .units import UnitDecoder
from .vocoder import CompactUnitVocoder


@dataclass(frozen=True)
class SeamTranslationOutput:
    semantic_logits: Tensor
    unit_logits: Tensor
    predicted_durations: Tensor
    waveform: Tensor


class SeamTranslationModel(nn.Module):
    """Trainable compact UnitY-style English-speech-to-Hindi-speech model."""

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
        self.unit_decoder = UnitDecoder(
            semantic_vocab_size=config.semantic_vocab_size,
            unit_vocab_size=config.unit_vocab_size,
            dim=config.unit_dim,
            layers=config.unit_layers,
            heads=config.unit_heads,
            ffn_dim=config.unit_ffn_dim,
            dropout=config.dropout,
            pad_token_id=config.pad_token_id,
        )
        self.vocoder = CompactUnitVocoder(config.unit_vocab_size, config.vocoder_channels)

    def encode(self, waveform: Tensor, waveform_lengths: Tensor, *,
               left_context: int | None = None,
               right_context: int | None = None) -> tuple[Tensor, Tensor]:
        features, feature_lengths = self.frontend(waveform, waveform_lengths)
        hidden, encoded_lengths = self.subsampler(features, feature_lengths)
        return self.encoder(
            hidden,
            encoded_lengths,
            left_context=left_context,
            right_context=right_context,
        )

    def forward(self, waveform: Tensor, waveform_lengths: Tensor, semantic_input_ids: Tensor,
                *, unit_durations: Tensor | None = None) -> SeamTranslationOutput:
        memory, memory_padding = self.encode(waveform, waveform_lengths)
        semantic_logits = self.semantic_decoder(
            semantic_input_ids, memory, memory_padding
        )
        units = self.unit_decoder(semantic_input_ids, unit_durations)
        unit_ids = units.logits.argmax(dim=-1)
        generated_waveform = self.vocoder(unit_ids)
        return SeamTranslationOutput(
            semantic_logits=semantic_logits,
            unit_logits=units.logits,
            predicted_durations=units.durations,
            waveform=generated_waveform,
        )

    @torch.inference_mode()
    def generate(self, waveform: Tensor, waveform_lengths: Tensor) -> tuple[Tensor, Tensor]:
        was_training = self.training
        self.eval()
        memory, memory_padding = self.encode(waveform, waveform_lengths)
        semantic_tokens = self.semantic_decoder.generate(
            memory,
            memory_padding,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
        )
        units = self.unit_decoder(semantic_tokens)
        generated_waveform = self.vocoder(units.logits.argmax(dim=-1))
        self.train(was_training)
        return semantic_tokens, generated_waveform

    def reset_state(self) -> None:
        """Erase all mutable state associated with an utterance."""
        self.encoder.reset_state()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
