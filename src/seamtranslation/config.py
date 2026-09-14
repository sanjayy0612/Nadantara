from dataclasses import dataclass

SUPPORTED_LANGUAGES = frozenset({"en", "hi"})


@dataclass(frozen=True, slots=True)
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    chunk_ms: int = 320

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.chunk_ms <= 0:
            raise ValueError("sample_rate and chunk_ms must be positive")
        if self.channels != 1 or self.sample_width_bytes != 2:
            raise ValueError("only mono signed 16-bit PCM audio is supported")
        if self.sample_rate * self.chunk_ms % 1_000:
            raise ValueError("chunk_ms must produce a whole number of samples")

    @property
    def samples_per_chunk(self) -> int:
        return self.sample_rate * self.chunk_ms // 1_000

    @property
    def bytes_per_chunk(self) -> int:
        return self.samples_per_chunk * self.channels * self.sample_width_bytes


@dataclass(frozen=True, slots=True)
class TranslationConfig:
    source_language: str = "en"
    target_language: str = "hi"

    def __post_init__(self) -> None:
        if self.source_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported source language: {self.source_language}")
        if self.target_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported target language: {self.target_language}")
        if self.source_language == self.target_language:
            raise ValueError("source and target languages must differ")
