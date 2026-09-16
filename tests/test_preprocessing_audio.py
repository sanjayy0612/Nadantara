from io import BytesIO
import wave

import numpy as np
import pytest

from seamtranslation.preprocessing import AudioSamples, decode_wav, encode_wav, resample


def _wav_bytes(sample_rate: int, samples: np.ndarray, *, channels: int = 1) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples.astype("<i2").tobytes())
    return output.getvalue()


def test_decode_resample_encode_to_model_contract() -> None:
    source_rate = 48_000
    time = np.arange(source_rate, dtype=np.float64) / source_rate
    pcm = np.rint(np.sin(2 * np.pi * 440 * time) * 16_000).astype("<i2")

    decoded = decode_wav(_wav_bytes(source_rate, pcm))
    converted = resample(decoded)
    result = decode_wav(encode_wav(converted))

    assert decoded.sample_rate == source_rate
    assert converted.sample_rate == 16_000
    assert len(converted.waveform) == 16_000
    assert result.sample_rate == 16_000
    assert result.waveform.dtype == np.float32
    assert np.max(np.abs(result.waveform)) == pytest.approx(16_000 / 32_768, abs=0.002)


def test_decode_rejects_stereo() -> None:
    stereo = np.zeros(20, dtype="<i2")
    with pytest.raises(ValueError, match="WAV format mismatch"):
        decode_wav(_wav_bytes(48_000, stereo, channels=2))


def test_audio_samples_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="one-dimensional float32"):
        AudioSamples(np.zeros((1, 4), dtype=np.float32), 16_000)
