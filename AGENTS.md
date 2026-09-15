# SeamTranslation contributor instructions

These instructions apply to the entire repository.

## Mission

Build a compact, offline English-speech-to-Hindi-speech translator for constrained
hardware. The primary development machine is an 8 GB Apple M2 MacBook Air. The second
deployment target is an 8 GB Jetson Orin Nano. The product interface is audio in and
audio out; internal Hindi semantic tokens are allowed and expected.

## Architectural contract

The selected model family is a compact UnitY-style pipeline:

```text
16 kHz English waveform
  → log-mel frontend and temporal subsampling
  → chunk-capable Conformer encoder
  → utterance-scoped Hindi semantic decoder
  → non-autoregressive Hindi speech-unit decoder
  → compact unit vocoder
  → Hindi waveform
```

Read `docs/product.md`, `docs/architecture.md`, `docs/model-spec.md`, and
`docs/evaluation.md` before changing model behavior. Consult
`docs/architecture-study.md` when reconsidering an architectural decision.

Do not silently replace this design with an ASR→MT→TTS cascade, decoder-only speech LLM,
direct waveform model, or MoE. Such systems may be implemented as measured baselines or
experiments, but changing the selected architecture requires an explicit decision-record
update with evidence.

## Non-negotiable behavior

- Initial direction is English (`en`) speech → Hindi (`hi`) speech only.
- Boundary audio is mono signed 16-bit PCM at 16 kHz.
- No network service is required during deployed inference.
- Do not preserve conversation history between utterances.
- Temporary encoder, decoder KV, unit, and vocoder state may exist only for the current
  utterance and must be reset after completion or cancellation.
- Keep batch-1 inference and bounded memory as first-class constraints.
- Prefer portable PyTorch operations that can be exported to Core ML and ONNX/TensorRT.
- Keep backend-specific code behind explicit interfaces.
- Never claim latency, memory, quality, FLOPS, or energy results without a reproducible
  measurement artifact.

## Engineering workflow

Before implementation:

1. Inspect the affected contracts, tests, and relevant design document.
2. State the component input/output shapes and state lifetime.
3. Establish a baseline for optimization work.

While implementing:

- Make one focused change at a time.
- Avoid dependencies that are not needed by the current executable slice.
- Keep teacher-only packages and checkpoints out of runtime dependencies.
- Keep datasets, checkpoints, generated audio, and benchmark artifacts out of Git.
- Record dataset source, version, license, checksum, and split-generation procedure.
- Treat synthetic and pseudo-labelled samples as such in manifests and reports.

After implementation:

```bash
PYTHONPATH=src python3 -m pytest -q
python3 -m compileall -q src tests
git diff --check
```

Run Ruff when installed. Model changes also require shape, reset-state, deterministic
smoke, and serialization tests. Performance changes require before/after measurements on
the same device, power mode, input manifest, precision, and runtime version.

## Git discipline

- Create one local commit per logical change after verification.
- Use imperative Conventional Commit messages such as `feat: add log-mel frontend`.
- Do not combine dataset acquisition, model behavior, optimization, and documentation in
  one commit unless they are inseparable.
- Do not commit or discard unrelated user changes.
- Do not push unless explicitly requested.

## Definition of done

Code existing in the repository is not enough. A slice is complete only when its exit
criteria in `docs/implementation-plan.md` pass, tests cover the affected contracts, and
the observed behavior is recorded without overstating results.
