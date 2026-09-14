from collections.abc import Iterable, Iterator

from .audio import AudioChunk
from .config import TranslationConfig
from .contracts import SpeechEncoder, TranslationCore, UnitDecoder, Vocoder


class SpeechToSpeechPipeline:
    def __init__(self, encoder: SpeechEncoder, translator: TranslationCore,
                 unit_decoder: UnitDecoder, vocoder: Vocoder) -> None:
        self.encoder = encoder
        self.translator = translator
        self.unit_decoder = unit_decoder
        self.vocoder = vocoder

    def translate(self, chunks: Iterable[AudioChunk], config: TranslationConfig
                  ) -> Iterator[AudioChunk]:
        for chunk in chunks:
            encoded = self.encoder.encode(chunk)
            semantic = self.translator.translate(
                encoded, source_language=config.source_language,
                target_language=config.target_language)
            units = self.unit_decoder.decode(semantic, target_language=config.target_language)
            yield self.vocoder.synthesize(units)
