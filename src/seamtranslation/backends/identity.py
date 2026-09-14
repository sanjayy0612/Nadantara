from collections.abc import Iterable

from ..audio import AudioChunk


def passthrough(chunks: Iterable[AudioChunk]) -> list[AudioChunk]:
    """Return chunks unchanged to validate audio transport only."""
    return list(chunks)
