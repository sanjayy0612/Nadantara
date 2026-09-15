from __future__ import annotations

from torch import Tensor, nn


class CompactUnitVocoder(nn.Module):
    """Small unit-conditioned waveform generator with 320× temporal upsampling."""

    def __init__(self, unit_vocab_size: int, channels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(unit_vocab_size, channels)
        rates = (5, 4, 4, 2, 2)
        blocks: list[nn.Module] = []
        current = channels
        for rate in rates:
            following = max(current // 2, 16)
            blocks.extend(
                (
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(
                        current,
                        following,
                        kernel_size=rate * 2,
                        stride=rate,
                        padding=(rate + 1) // 2,
                        output_padding=rate % 2,
                    ),
                )
            )
            current = following
        blocks.extend((nn.LeakyReLU(0.1), nn.Conv1d(current, 1, kernel_size=7, padding=3)))
        self.generator = nn.Sequential(*blocks)

    def forward(self, unit_ids: Tensor) -> Tensor:
        if unit_ids.ndim != 2:
            raise ValueError("unit_ids must have shape [batch, unit_steps]")
        hidden = self.embedding(unit_ids).transpose(1, 2)
        return self.generator(hidden).squeeze(1).tanh()
