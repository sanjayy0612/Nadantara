# Translation quality audit

Findings from auditing the distillation pipeline and 2.69M `SpeechSemanticModel`
after the first Student A / Student B training run, plus what would move
translation quality forward.

## Held-out benchmark results (100-example, never used in training)

| | Student A (Whisper→IndicTrans2) | Student B (SeamlessM4T-v2) |
|---|---|---|
| Held-out val loss | 5.193 | 5.181 |
| chrF2 | 8.745 | 9.057 |
| BLEU | 0.756 | 0.450 |
| EOS rate | 0.980 | 0.940 |
| Exact match | 0.000 | 0.000 |
| Adjacent repetition rate | 0.537 | 0.651 |

Output is fluent-looking Hindi (real words, valid grammar fragments) but
semantically disconnected from the source and heavily repetitive
(`जो जो जो...`, `उसके उसके...`). Both students behave the same way, so the
cause is shared infrastructure, not a per-teacher issue.

## After the no-repeat-ngram fix (same checkpoints, no retraining)

Root cause #1 below was implemented (`no_repeat_ngram_size=3` in
`SemanticDecoder.generate()`) and re-evaluated on the same `best.pt`
checkpoints, isolating the decoding fix from any training change:

| | Student A — before → after | Student B — before → after |
|---|---|---|
| chrF2 | 8.745 → **11.449** | 9.057 → **14.924** |
| BLEU | 0.756 → **1.365** | 0.450 → **1.135** |
| EOS rate | 0.980 → **1.000** | 0.940 → **0.990** |
| Exact match | 0.000 → 0.000 | 0.000 → 0.000 |
| Adjacent repetition rate | 0.537 → **0.193** | 0.651 → **0.194** |

Repetition dropped ~3x and chrF2/BLEU both improved meaningfully, confirming
decoding was a real, measurable contributor — not just cosmetic. Still not
usable translation: chrF2 in the low teens, zero exact matches, output still
semantically disconnected from the source, and Student B still shows
sub-word-level stutter (e.g. repeated single characters) that a 3-gram block
doesn't catch. Root causes #2–6 are unaddressed and still needed for actual
usable quality.

## Root causes, ranked by expected impact

### 1. Greedy decoding with no repetition control — fixed
`src/seamtranslation/models/semantic.py:53` `SemanticDecoder.generate()` was
pure `argmax` per step: no repetition penalty, no no-repeat-ngram blocking,
no beam search, no length penalty. This was the direct cause of the
`जो जो जो` / `उसके उसके` loops — a known degenerate mode for undertrained
seq2seq decoders under greedy search. This was an **inference-time** fix: it
doesn't touch trained weights or violate the "same architecture/config for
both students" constraint, so it was safe to change without retraining.

**Status: implemented.** Added `no_repeat_ngram_size=3` blocking (see
`_ban_repeated_ngrams` in the same file). Results above under "After the
no-repeat-ngram fix" — real improvement, not sufficient alone.

### 2. Very little training signal
- Pseudo-pretrain: 2000 steps × batch 9 ≈ 18k samples seen over 9979 unique
  examples (~1.8 epochs).
- Real fine-tune: 1000 steps × batch 9 ≈ 9k samples over 1736 unique
  examples (~5.2 epochs).

Both stages are short relative to a from-scratch seq2seq model with no
pretrained initialization. Train/validation loss was still trending down at
the last logged step in both runs (5.27 → still falling), meaning the model
had not converged when training stopped — `max_steps` cut it off, not a
plateau.

**Fix:** increase `max_steps` for both stages (start by doubling) and watch
`generation_chrf2`/`exact_match_rate` in the periodic eval rather than just
loss, since loss alone doesn't reveal repetition collapse.

### 3. No learning-rate schedule
`train/train_semantic.py` uses a flat `AdamW(lr=2e-4)` for the entire run —
no warmup, no decay (`train/train_semantic.py:62`). A short linear warmup
(e.g. 100–200 steps) plus cosine or linear decay toward the end of each stage
is standard for transformer decoders this size and usually reduces the kind
of instability that pushes a model into repetition loops.

### 4. Pseudo-label quality ceiling
Pseudo-pretrain labels come from imperfect automatic pipelines: Whisper ASR
→ IndicTrans2 MT (Student A, two compounding error sources) and
SeamlessM4T-v2 zero-shot S2TT (Student B, one source but zero-shot). Neither
teacher is a gold reference. The pretrain stage is teaching the model
Hindi surface statistics from these pseudo-labels before it ever sees the
1736 real human-labeled examples in fine-tune — if teacher label quality is
low, pretrain can actively teach wrong content, which fine-tune then has to
undo with 5x less data than it was pretrained on.

**Check before spending more compute:** sample ~20 pseudo-labels from each
teacher's `pseudo_train_filtered.jsonl` against the source English text by
hand. If translations are frequently wrong (not just imperfect), the
pretrain stage may be hurting more than helping.

### 5. 2,000-token Hindi vocabulary is small for open-domain text
`model_registry/tokenizers/hindi_2k.json` — a 2K unigram SentencePiece vocab
fragments many Hindi words into several pieces, which lengthens sequences
the decoder must get exactly right and gives repetition more places to
start. A `hindi_8k` tokenizer already exists in the registry
(`train/README.md` references it) — worth an ablation if steps 1–4 don't
close the gap, though this changes the architecture contract shared between
students, so it's a bigger, coordinated change, not a quick fix.

### 6. No label smoothing
`semantic_loss` (`src/seamtranslation/training/semantic.py:105`) is plain
`F.cross_entropy` with no `label_smoothing`. Cheap to add
(`label_smoothing=0.1` is a common default) and known to reduce
overconfident repetition in small transformer decoders.

## What audited clean

- `LogMelFrontend` / `ConvSubsampler` (`src/seamtranslation/models/frontend.py`):
  mel filterbank math, STFT framing, and the length-tracking through the two
  stride-2 convolutions all check out.
- `write_training_manifests` audio-path bug (found and fixed earlier this
  session, `experiments/distillation/colab_common.py:865`) — unrelated to
  quality, but was blocking Student B's real fine-tune stage entirely before
  the fix.
- Real fine-tune manifest correctly uses human reference text
  (`real_data/manifest.jsonl` → `target_text`), not teacher pseudo-labels —
  the common-ID gate and human-vs-pseudo separation are implemented
  correctly.

## Suggested order of operations

1. ~~Add no-repeat-ngram blocking to `SemanticDecoder.generate()`~~ — done,
   see results above. Repetition down ~3x, chrF2/BLEU up, still far from
   usable.
2. Add label smoothing (`label_smoothing=0.1`) — cheap, one-line change for
   the next training run.
3. Add LR warmup + decay.
4. Increase `max_steps` for both stages (2-4x) and re-train, watching
   `generation_chrf2` / `exact_match_rate`, not just loss.
5. Manually audit ~20 pseudo-label samples per teacher before deciding
   whether to trust pretrain as-is.
6. Only after 1-5: consider the 8K tokenizer as a coordinated,
   both-students change.
