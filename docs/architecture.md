# Architecture decision: compact UnitY-style speech translation

The evidence and hardware analysis supporting this decision are recorded in
[architecture-study.md](architecture-study.md).

## Objective and design

Translate English speech directly into Hindi speech on low-power ARM hardware. A
chunk-capable speech encoder produces acoustic representations; an autoregressive,
utterance-scoped translation decoder produces Hindi semantic tokens; a
non-autoregressive decoder predicts discrete speech units; and a lightweight vocoder
produces waveform samples.
Text is allowed as auxiliary training supervision but is not exposed during inference.

This retains the useful Seamless/UnitY decomposition without massively multilingual
capacity, permits independent distillation and profiling, and keeps operations portable
between PyTorch MPS, Core ML and TensorRT.

## Initial non-goals

- Training a multi-billion-parameter model from random initialization
- Arbitrary languages, speakers, voice cloning, or full-duplex conversation
- MoE before measuring a dense baseline

## Invariants

- Boundary audio is mono signed 16-bit PCM at 16 kHz.
- Source and target languages are explicit ISO 639-1 identifiers.
- Streaming state belongs to model components, not audio transport.
- All utterance state, including temporary decoder KV state, is erased after output.
- Backend-specific acceleration stays behind component interfaces.
- Every optimization is compared against a measured baseline.
