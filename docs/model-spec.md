# Compact UnitY model specification

This document defines the first model we intend to implement. Dimensions are initial
budgets and may change only through measured profiling or training evidence.

## End-to-end graph

```text
PCM waveform [B, samples]
  ↓ AudioFrontend
log-mel features [B, frames, 80]
  ↓ 4× convolutional subsampler
subsampled features [B, about frames/4, 384]
  ↓ SpeechEncoder
encoded speech [B, source_steps, 384]
  ↓ SemanticDecoder
Hindi token IDs [B, target_tokens]
  ↓ UnitDecoder
Hindi speech-unit IDs [B, unit_steps]
  ↓ Vocoder
waveform [B, output_samples]
```

`B` is one during deployed inference. Training may use larger or accumulated batches.

## Audio frontend

- Sample rate: 16,000 Hz.
- Window: 25 ms / 400 samples.
- Hop: 10 ms / 160 samples.
- Features: 80 log-mel bins.
- Subsampling: two stride-2 convolutional layers, 4× total.
- Output width: 384.

Feature extraction must have a numerically checked reference implementation. Platform
accelerators may replace it only when their output remains within a declared tolerance.

## Speech encoder

- Family: chunk-capable Conformer.
- Blocks: 12.
- Width: 384.
- Heads: 6; head width: 64.
- Feed-forward width: 1,536.
- Convolution kernel: initial value 15.
- Positional method: exportable relative or rotary scheme chosen by prototype evidence.
- Parameter budget: 35–55M.

Offline mode uses the same layers with full utterance visibility. Streaming mode uses
bounded left context and configurable look-ahead. State must have an explicit reset API.

## Hindi semantic decoder

- Family: autoregressive encoder-decoder Transformer.
- Blocks: 6.
- Width: 384.
- Heads: 6.
- Feed-forward width: 1,536.
- Vocabulary: Hindi-focused SentencePiece, initial size 8,000.
- Decoding: greedy baseline, then beam size 2 only if quality gain justifies latency.
- Maximum output: 128 semantic tokens per utterance initially.
- Parameter budget: 20–35M.

Its KV cache is temporary computational state, not conversation memory. It is allocated
for one utterance and destroyed afterward. At the initial limits its self-attention K/V
payload is roughly 1.1 MiB in FP16.

## Hindi speech-unit decoder

- Family: non-autoregressive text/character-to-unit model.
- Blocks: 4.
- Width: 256.
- Explicit duration prediction and upsampling.
- Parameter budget: 10–20M.
- Unit vocabulary: intentionally undecided until tokenizer/vocoder candidates are
  measured for Hindi intelligibility, sequence rate, codebook size, and licensing.

## Vocoder

- Family: compact unit-conditioned HiFi-GAN-style generator.
- Initial precision: FP16.
- Parameter budget: 10–20M.
- Output rate: 16 or 24 kHz, selected with the unit/vocoder checkpoint.

The product boundary will resample explicitly if model output differs from the 16 kHz
internal transport rate. Vocoder quantization requires listening and intelligibility
evaluation; model size alone is not sufficient evidence.

## Whole-model constraints

| Property | Initial constraint |
|---|---:|
| Parameters | 60–130M |
| FP16 weights | 120–260 MB |
| INT8 weights | 60–130 MB plus scales/metadata |
| Peak inference memory | ≤2 GB target |
| Deployment batch | 1 |
| Initial utterance | ≤15 seconds |

## Precision policy

Establish correctness in FP32/FP16. Then test INT8 per component. Linear and convolution
weights are likely candidates; normalization, softmax, output projections and the
vocoder remain FP16 when quantization harms stability or speech quality. Sparse speedup
is never assumed without device measurement.

The initial default PyTorch configuration contains 65,247,994 parameters. This implies
approximately 124.5 MiB of raw FP16 weights or 62.2 MiB of raw INT8 weights. It does not
include activations, runtime workspaces, allocator overhead, caches, or quantization
metadata and therefore is not a runtime-memory benchmark.
