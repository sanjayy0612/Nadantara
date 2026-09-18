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

    Scores use SacreBLEU's 0-100 scale. Adjacent repetition is the fraction of
    token transitions where the next token is identical to the previous token.
    """
    size = len(hypotheses)
    if not size:
        raise ValueError("at least one generated translation is required")
    if not (len(references) == len(token_ids) == len(ended_with_eos) == size):
        raise ValueError("hypotheses, references, token IDs and EOS flags must align")

    adjacent_repetitions = 0
    token_transitions = 0
    generated_tokens = 0
    for sentence in token_ids:
        generated_tokens += len(sentence)
        token_transitions += max(0, len(sentence) - 1)
        adjacent_repetitions += sum(
            current == previous for previous, current in zip(sentence, sentence[1:])
        )

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
        "exact_match_rate": sum(
            hypothesis.strip() == reference.strip()
            for hypothesis, reference in zip(hypotheses, references, strict=True)
        )
        / size,
        "adjacent_repetition_rate": (
            adjacent_repetitions / token_transitions if token_transitions else 0.0
        ),
        "generated_tokens": generated_tokens,
    }
