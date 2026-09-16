"""Decode, validate, and resample training audio to the model boundary format."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import wave

import numpy as np
from scipy.signal import resample_poly

from seamtranslation.config import AudioConfig


@dataclass(frozen=True, slots=True)
class AudioSamples:
    """Mono floating-point audio normalized to the interval [-1, 1]."""

    waveform: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.waveform.dtype != np.float32 or self.waveform.ndim != 1:
            raise ValueError("waveform must be a one-dimensional float32 array")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")


def decode_wav(payload: bytes) -> AudioSamples:
    """Decode signed 16-bit PCM WAV bytes and downmix channels to mono."""

    try:
        with wave.open(BytesIO(payload), "rb") as stream:
            channels = stream.getnchannels()
            observed = (stream.getsampwidth(), stream.getcomptype())
            expected = (2, "NONE")
            if channels <= 0 or observed != expected:
                raise ValueError(
                    "WAV format mismatch: expected signed 16-bit PCM with at least one "
                    f"channel, got channels={channels}, format={observed}"
                )
            sample_rate = stream.getframerate()
            frames = stream.readframes(stream.getnframes())
    except (EOFError, wave.Error) as exc:
        raise ValueError("invalid WAV payload") from exc

    pcm = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    waveform = pcm.astype(np.float32).mean(axis=1) / 32_768.0
    return AudioSamples(waveform=waveform, sample_rate=sample_rate)


def resample(audio: AudioSamples, target_rate: int = 16_000) -> AudioSamples:
    """Band-limited polyphase resampling with deterministic output length."""

    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if audio.sample_rate == target_rate:
        return audio

    divisor = np.gcd(audio.sample_rate, target_rate)
    waveform = resample_poly(
        audio.waveform,
        up=target_rate // divisor,
        down=audio.sample_rate // divisor,
    ).astype(np.float32, copy=False)
    waveform = np.clip(waveform, -1.0, 1.0)
    return AudioSamples(waveform=waveform, sample_rate=target_rate)


def encode_wav(audio: AudioSamples, config: AudioConfig | None = None) -> bytes:
    """Encode samples as mono signed 16-bit PCM WAV bytes."""

    config = config or AudioConfig(sample_rate=audio.sample_rate)
    if audio.sample_rate != config.sample_rate:
        raise ValueError(
            f"sample-rate mismatch: audio={audio.sample_rate}, config={config.sample_rate}"
        )

    pcm = np.rint(np.clip(audio.waveform, -1.0, 1.0) * 32_767.0).astype("<i2")
    output = BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(config.channels)
        stream.setsampwidth(config.sample_width_bytes)
        stream.setframerate(config.sample_rate)
        stream.writeframes(pcm.tobytes())
    return output.getvalue()
