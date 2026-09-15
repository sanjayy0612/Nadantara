# Implementation plan

This is one product built through verifiable vertical slices. Slice numbers describe
dependency order, not disposable product versions.

## Slice 0 — system foundation

Status: complete.

- Audio configuration and deterministic WAV chunking.
- Model component protocols.
- Runtime selection for MPS, CUDA, or CPU.
- Identity audio transport and tests.

Exit evidence: byte-exact WAV round trip and passing unit tests.

## Slice 1 — reproducible baselines and manifests

- Define versioned evaluation-manifest and benchmark-result schemas.
- Audit candidate datasets for direction, provenance, license, speaker overlap and
  synthetic content.
- Add a local-only checkpoint registry with hashes.
- Benchmark SeamlessM4T and the cascade teacher on a small fixed sample.

Exit evidence: one command reproduces the baseline report; no downloaded data or weights
are committed.

## Slice 2 — audio frontend and encoder contract

- Implement reference log-mel extraction and 4× subsampling.
- Add tensor-shape and numerical-reference tests.
- Implement the small Conformer configuration and state-reset API.
- Validate full-utterance forward execution before training.

Exit evidence: deterministic shapes, bounded state, parameter-count report, and an
exportability smoke test.

## Slice 3 — English speech to Hindi semantics

- Train or import the 8K Hindi tokenizer with recorded provenance.
- Implement the semantic decoder and temporary KV cache.
- Pretrain/distil with English–Hindi text and train on speech-translation data.
- Compare generated Hindi with the cascade baseline.

Exit evidence: a frozen checkpoint beats a declared minimum quality threshold on an
unseen manifest and resets cleanly between utterances.

## Slice 4 — Hindi semantics to speech units

- Benchmark Hindi-capable unit-tokenizer and vocoder candidates.
- Freeze the selected unit definition and target-speech rate.
- Implement duration prediction and non-autoregressive unit generation.
- Train with real and clearly labelled synthetic Hindi speech.

Exit evidence: Hindi units produce intelligible audio and the choice is supported by
quality, speed, size, and license evidence.

## Slice 5 — complete compact speech-to-speech model

- Connect encoder, semantic decoder, unit decoder, and vocoder.
- Add cancellation and guaranteed utterance-state cleanup.
- Fine-tune connected components where data and compute permit.
- Compare against both fixed baselines.

Exit evidence: held-out English audio produces intelligible, semantically faithful Hindi
audio in one command.

## Slice 6 — M2 optimization

- Profile the FP16 PyTorch/MPS baseline.
- Export supported graphs to Core ML and inspect device placement.
- Apply component-wise compression only after quality measurement.
- Optimize the measured bottleneck.

Exit evidence: peak inference memory ≤2 GB and a reproducible latency/power report.

## Slice 7 — Jetson deployment

- Export ONNX with bounded shapes and explicit state tensors.
- Build TensorRT 10.x engines on the target JetPack/`sm_87` environment.
- Establish FP16, then test INT8 and optional 2:4 sparsity.
- Implement production orchestration with persistent execution contexts and buffers.

Exit evidence: sustained benchmark in both normal and Super power modes with no hidden
CPU fallback.

## Slice 8 — streaming output

- Activate bounded encoder context and look-ahead.
- Add endpoint detection and a conservative read/write policy.
- Generate units and waveform incrementally without boundary artifacts.
- Include all buffering in time-to-first-audio.

Exit evidence: real-time factor below 1.0 on the selected device, bounded memory, and an
explicit quality-versus-latency curve.

## Immediate next change

Implement the manifest and benchmark schemas before downloading any model or dataset.
That gives every later experiment a reproducible identity and prevents untracked
checkpoint/data decisions.
