from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import wave

from .config import AudioConfig


@dataclass(frozen=True, slots=True)
class AudioChunk:
    pcm_s16le: bytes
    start_sample: int
    is_final: bool


def read_wav_chunks(path: Path, config: AudioConfig) -> Iterator[AudioChunk]:
    with wave.open(str(path), "rb") as stream:
        observed = (stream.getframerate(), stream.getnchannels(), stream.getsampwidth())
        expected = (config.sample_rate, config.channels, config.sample_width_bytes)
        if observed != expected:
            raise ValueError(f"WAV format mismatch: expected {expected}, got {observed}")
        start_sample = 0
        pending = stream.readframes(config.samples_per_chunk)
        while pending:
            following = stream.readframes(config.samples_per_chunk)
            yield AudioChunk(pending, start_sample, is_final=not following)
            start_sample += len(pending) // config.sample_width_bytes
            pending = following


def write_wav(path: Path, chunks: Iterator[AudioChunk], config: AudioConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(config.channels)
        stream.setsampwidth(config.sample_width_bytes)
        stream.setframerate(config.sample_rate)
        for chunk in chunks:
            stream.writeframesraw(chunk.pcm_s16le)
