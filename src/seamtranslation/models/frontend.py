from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> Tensor:
    def hz_to_mel(value: Tensor) -> Tensor:
        return 2_595.0 * torch.log10(1.0 + value / 700.0)

    def mel_to_hz(value: Tensor) -> Tensor:
        return 700.0 * (torch.pow(10.0, value / 2_595.0) - 1.0)

    min_mel = hz_to_mel(torch.tensor(0.0))
    max_mel = hz_to_mel(torch.tensor(sample_rate / 2.0))
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_hz = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    rising = (fft_hz - lower) / (center - lower).clamp_min(1e-8)
    falling = (upper - fft_hz) / (upper - center).clamp_min(1e-8)
    return torch.minimum(rising, falling).clamp_min(0.0)


class LogMelFrontend(nn.Module):
    """Differentiable 16 kHz waveform-to-log-mel frontend."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.register_buffer("window", torch.hann_window(config.n_fft), persistent=False)
        self.register_buffer(
            "mel_filters",
            _mel_filterbank(config.sample_rate, config.n_fft, config.feature_bins),
            persistent=True,
        )

    def forward(self, waveform: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        if lengths.shape != (waveform.shape[0],):
            raise ValueError("lengths must have shape [batch]")
        if waveform.shape[1] < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - waveform.shape[1]))
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spectrum.abs().square()
        features = torch.einsum("mf,bft->btm", self.mel_filters, power)
        features = torch.log(features.clamp_min(1e-5))
        frame_lengths = ((lengths.clamp_min(self.n_fft) - self.n_fft) // self.hop_length + 1)
        return features, frame_lengths.clamp_max(features.shape[1])


class ConvSubsampler(nn.Module):
    """Reduce acoustic sequence length by four before attention."""

    def __init__(self, feature_bins: int, model_dim: int) -> None:
        super().__init__()
        channels = model_dim // 4
        if channels == 0:
            raise ValueError("model_dim must be at least 4")
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        reduced_bins = math.ceil(math.ceil(feature_bins / 2) / 2)
        self.projection = nn.Linear(channels * reduced_bins, model_dim)

    def forward(self, features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.convolutions(features.unsqueeze(1)).transpose(1, 2)
        batch, steps, channels, bins = hidden.shape
        hidden = self.projection(hidden.reshape(batch, steps, channels * bins))
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return hidden, lengths.clamp_max(steps)
