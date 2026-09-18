"""Quality metrics for generated Hindi semantic translations."""

from __future__ import annotations

from collections.abc import Sequence

import sacrebleu


def semantic_generation_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str],
    token_ids: Sequence[Sequence[int]],
    ended_with_eos: Sequence[bool],
) -> dict[str, float | int]:
    """Return corpus translation and decoding-health metrics.

    Scores use SacreBLEU's 0-100 scale. Repetition is the fraction of generated
    token positions that repeat a token already emitted in the same sentence.
    """
    size = len(hypotheses)
    if not size:
        raise ValueError("at least one generated translation is required")
    if not (len(references) == len(token_ids) == len(ended_with_eos) == size):
        raise ValueError("hypotheses, references, token IDs and EOS flags must align")

    repeated = 0
    generated = 0
    for sentence in token_ids:
        generated += len(sentence)
        repeated += len(sentence) - len(set(sentence))

    return {
        "examples": size,
        "chrf2": float(
            sacrebleu.corpus_chrf(hypotheses, [list(references)], word_order=2).score
        ),
        "bleu": float(
            sacrebleu.corpus_bleu(
                hypotheses, [list(references)], tokenize="intl", use_effective_order=True
            ).score
        ),
        "eos_rate": sum(ended_with_eos) / size,
        "empty_rate": sum(not text.strip() for text in hypotheses) / size,
        "repeated_token_rate": repeated / generated if generated else 0.0,
        "generated_tokens": generated,
    }
