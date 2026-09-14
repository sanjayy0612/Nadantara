from dataclasses import dataclass
from typing import Protocol, TypeAlias

from .audio import AudioChunk

FloatVector: TypeAlias = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EncodedSpeech:
    frames: tuple[FloatVector, ...]
    is_final: bool


@dataclass(frozen=True, slots=True)
class SemanticState:
    tokens: tuple[int, ...]
    is_final: bool


@dataclass(frozen=True, slots=True)
class SpeechUnits:
    unit_ids: tuple[int, ...]
    is_final: bool


class SpeechEncoder(Protocol):
    def encode(self, chunk: AudioChunk) -> EncodedSpeech: ...


class TranslationCore(Protocol):
    def translate(self, speech: EncodedSpeech, *, source_language: str,
                  target_language: str) -> SemanticState: ...


class UnitDecoder(Protocol):
    def decode(self, semantics: SemanticState, *, target_language: str) -> SpeechUnits: ...


class Vocoder(Protocol):
    def synthesize(self, units: SpeechUnits) -> AudioChunk: ...
