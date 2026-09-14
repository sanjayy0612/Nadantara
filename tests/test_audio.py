from pathlib import Path
import wave

import pytest

from seamtranslation.audio import read_wav_chunks, write_wav
from seamtranslation.config import AudioConfig


def make_wav(path: Path, frames: int, sample_rate: int = 16_000) -> bytes:
    payload = b"\x01\x00" * frames
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(sample_rate)
        stream.writeframes(payload)
    return payload


def test_chunking_and_round_trip(tmp_path: Path) -> None:
    config = AudioConfig()
    source, target = tmp_path / "in.wav", tmp_path / "out.wav"
    expected = make_wav(source, config.samples_per_chunk * 2 + 7)
    chunks = list(read_wav_chunks(source, config))
    assert [c.start_sample for c in chunks] == [0, 5_120, 10_240]
    assert [c.is_final for c in chunks] == [False, False, True]
    write_wav(target, iter(chunks), config)
    with wave.open(str(target), "rb") as stream:
        assert stream.readframes(stream.getnframes()) == expected


def test_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    make_wav(source, 10, 8_000)
    with pytest.raises(ValueError, match="WAV format mismatch"):
        list(read_wav_chunks(source, AudioConfig()))
