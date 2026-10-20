# Local baseline continuation — 2026-10-01

Continued the existing 2,688,352-parameter real-data baseline from step 2900 to
3900 on the local MPS accelerator. Used the same model, Hindi-2K tokenizer,
1,736 training examples and 264 validation examples. The new learning-rate
schedule warmed up over 50 steps to 0.0002 and decayed to 0.00002.
Training took 821.3 seconds. Original artifacts were left intact; new weights
and metadata are under `train/runs/semantic-2k-local-continuation/` (ignored).

## Full validation evaluation

All three checkpoints used the current decoder and the same 264 examples.
Shuffled audio used seed 17 with no example retaining its own audio.

| Checkpoint | Step | chrF2 | BLEU | Adjacent repetition | Correct minus shuffled chrF2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2900 | 14.201 | 1.141 | 15.44% | 2.006 |
| Continuation best validation loss | 3800 | 13.907 | 1.138 | 15.69% | 2.080 |
| Continuation final | 3900 | 13.781 | 1.197 | 16.03% | 2.354 |

Best validation loss decreased from 5.0811 to 4.9662, but neither tested
continuation checkpoint improved chrF2. The final checkpoint's small BLEU gain
does not establish a meaningful overall improvement. Generated samples remain
repetitive and omit source meaning. The audio-dependence gap remains small.

This bounded experiment does not support extending the same recipe blindly.
It does not establish model capacity as the bottleneck, prove the encoder is
broken, or rule out longer training under other conditions. Duration and schedule
changed together, so their individual effects cannot be separated here.

These are development-set measurements, not an independent final test. This
experiment used the real-data baseline, not the later Teacher A/B students.
The existing 16-example memorization record was inspected, not retrained.

## Artifacts and verification

- `baseline_audio_dependence.json`: baseline metrics and paired outputs.
- `continuation_best_audio_dependence.json`: step-3800 metrics and outputs.
- `continuation_checkpoint_audio_dependence.json`: step-3900 metrics and outputs.
- `memorization_record.json`: saved 16-example result and source location.
- JSON reports include checkpoint, manifest and tokenizer hashes.
- Full repository tests: 27 passed; `git diff --check` passed.

Next, inspect equivalent diagnostics for the distilled students if their
checkpoints are available. For a new training experiment, compare pretrained
speech initialization or a modest capacity increase under a fixed development
evaluation; neither alternative is validated by this run.
