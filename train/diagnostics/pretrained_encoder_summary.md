# Frozen pretrained speech encoder experiment — 2026-10-01

## Question and design

Does frozen pretrained speech representation improve Hindi translation under a
bounded adaptation budget, compared with the same randomly initialized frozen
encoder architecture?

Compared pretrained and random Whisper Tiny encoders at revision
`169d4a4341b33bc18d8881c4b69c2e104e1cc0af`. Both encoders have 8,208,384
parameters and remain frozen. Each condition trains an identical 384-to-128
adapter and Hindi decoder (1,115,904 trainable parameters), totaling 9,324,288
parameters. The original 2,688,352-parameter model is only a reference baseline;
its architecture and training budget differ from this experiment.

Both conditions use the original baseline's step-2900 Hindi decoder weights,
the same adapter initialization, seed 17, all 1,736 training examples, Hindi-2K
tokenizer, batch size 9, and 1,000 training steps. Learning rate warms up for
50 steps to 0.0002, then decays to 0.00002. Speech features are cached per
condition; only the adapter and decoder are optimized on the local MPS device.
The Hindi decoder receives speech features directly, not English transcripts.

Encoder preprocessing pads to Whisper's 30-second window and retains feature
positions corresponding to the actual audio duration. The maximum corpus
utterance is 19.92 seconds, so no source audio is truncated. Hindi target limits
match the existing baseline's 128-token policy.

## Results on all 264 development examples

| Model | chrF2 | BLEU | Adjacent repetition | Correct minus shuffled chrF2 |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 14.201 | 1.141 | 15.44% | 2.006 |
| Frozen random Whisper encoder | 13.929 | 0.915 | 20.69% | 0.811 |
| Frozen pretrained Whisper encoder | 13.784 | 0.956 | 18.28% | 1.386 |

Pretrained minus random chrF2 is -0.145. Pretraining reduces repetition relative
to the random control and slightly improves BLEU, but does not improve chrF2.
Neither condition establishes a useful translation-quality gain over the
original baseline. Generated examples remain repetitive and fail to preserve
source meaning. The correct-versus-shuffled audio gaps remain small.

## Interpretation and limits

This experiment does not support adopting a frozen Whisper encoder with this
adaptation recipe. It does not establish that pretraining in general is
ineffective: the encoder was not fine-tuned, the Hindi decoder still lacks
large-scale translation pretraining, the budget was bounded, and only one seed
was tested. No statistical significance claim is made for the small score
differences. This is development evaluation, not an independent test set.

The original baseline's successful 16-example memorization gate demonstrates
that it could learn a small training subset; it does not prove correct
generalization or adequate English-to-Hindi semantic learning.

Before another large run, isolate the translation stage using verified English
transcripts and inspect source/target alignment. A transcript-conditioned
translation control can help distinguish inadequate speech conditioning from
inadequate Hindi translation learning. That control has not been run here.

## Artifacts and verification

- Reusable experiment: `train/compare_pretrained_encoder.py`.
- Compact checked-in report: `train/diagnostics/pretrained_encoder_comparison.json`.
- Local heads, feature caches, full hypotheses/references, and hashes:
  `train/runs/frozen-whisper-comparison/` (ignored).
- Pinned source model and processor: `artifacts/pretrained/whisper-tiny/` (ignored).
- Data and inference stayed local; only public weights were downloaded.
- Tests: 29 passed, including adapter gradients and padding-mask correctness.
- `git diff --check` passed.

Source model details: https://huggingface.co/openai/whisper-tiny
