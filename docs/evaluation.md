# Evaluation contract

Optimization is accepted only when compared on the same manifest and hardware setup.

## Fixed comparisons

1. SeamlessM4T English→Hindi speech output as a large teacher/reference.
2. Whisper → IndicTrans2 → Hindi TTS as a strong modular baseline.
3. The compact UnitY-style student in FP16.
4. Quantized student variants, each compared with the FP16 student.

Teacher quality and licensing must be verified before checkpoints become project
defaults.

## Quality

- Source ASR transcript, when available, is evaluation metadata—not deployed output.
- Hindi semantic translation: chrF++ and COMET when the chosen language model supports
  Hindi reliably; report BLEU for comparison, not as the only metric.
- Speech translation: transcribe generated Hindi and report ASR-based translation
  metrics against the same references.
- Speech intelligibility and naturalness: a documented native-speaker listening sample.
- Error slices: names, numbers, negation, tense, gender, word order, accented English,
  code-switching, noise, and long utterances.

## Performance

Report at least:

- cold and warm time to first translated audio;
- end-to-end latency and real-time factor;
- p50 and p95 across the fixed manifest;
- peak resident/unified memory;
- model weight size and precision;
- CPU/GPU utilization and memory bandwidth where available;
- sustained power, temperature and throttling behavior.

Do not report component-only latency as end-to-end latency. Include capture buffering,
feature extraction, model execution, synthesis, resampling, and playback buffering.

## Reproducibility metadata

Every benchmark record must include:

- Git commit and configuration file;
- device model and memory;
- OS, Python, framework and accelerator runtime versions;
- Core ML compute-unit policy or Jetson power mode;
- model/checkpoint hashes;
- dataset manifest hash;
- warm-up count, repetitions, precision, and decoding settings.

## State-reset test

Run utterance A, then B, then B on a fresh process. The two outputs for B must agree
within the model's declared deterministic tolerance. This detects accidental leakage of
conversation or streaming state.
