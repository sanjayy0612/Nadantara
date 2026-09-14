import pytest

from seamtranslation.config import AudioConfig, TranslationConfig


def test_defaults() -> None:
    assert AudioConfig().samples_per_chunk == 5_120
    assert AudioConfig().bytes_per_chunk == 10_240
    assert TranslationConfig("en", "hi").target_language == "hi"


def test_languages_must_be_supported_and_different() -> None:
    with pytest.raises(ValueError, match="must differ"):
        TranslationConfig("hi", "hi")
    with pytest.raises(ValueError, match="unsupported source"):
        TranslationConfig("fr", "en")
