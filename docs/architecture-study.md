# English-to-Hindi edge speech translation: architecture study

Status: accepted direction, pending component benchmarks
Date: 2026-09-15

## Scope and review method

This is a narrative scoping review for one engineering question:

> Which English-speech-to-Hindi-speech architecture offers the strongest practical
> quality, latency, memory, data-efficiency and deployment trade-off on an 8 GB Apple M2
> and an 8 GB Jetson Orin Nano?

The review prioritizes peer-reviewed papers, original preprints, project repositories,
dataset publications and vendor documentation from 2019–2026. Results for other language
pairs are treated as architectural evidence, not as proof of English-to-Hindi quality.
Marketing TOPS/FLOPS are treated as theoretical ceilings, never application benchmarks.

## Decision

Build a compact, two-pass UnitY-style student:

```text
English waveform
  → log-mel frontend and 4× temporal subsampling
  → chunk-capable Conformer speech encoder
  → autoregressive Hindi semantic/subword decoder
  → non-autoregressive Hindi speech-unit decoder
  → compact unit-conditioned vocoder
  → Hindi waveform
```

The product boundary remains audio-to-audio. Hindi semantic tokens are an internal
representation and training target; they are not a separate user-visible service.

This is the best-supported compromise because UnitY's semantic first pass improved
translation quality by 2.5–4.2 ASR-BLEU over a single-pass speech-to-unit model while
also reporting 2.83× faster decoding. Discrete units were 2.51× faster to decode than
spectrogram prediction in the same study. [UnitY](https://arxiv.org/abs/2212.08055)

## Alternatives considered

| Method | Strength | Limitation for this project | Decision |
|---|---|---|---|
| ASR → MT → TTS | Best use of independently available data; easy to debug | Three deployed models, duplicated representations, accumulated latency and errors | Quality baseline and fallback |
| Direct speech → Hindi text → TTS | Removes English ASR text and one model boundary | TTS remains separately optimized; less joint speech-unit training | Strong implementation baseline |
| Direct speech → speech units | Simple inference path and no text dependency | Harder translation learning; loses the useful semantic bottleneck | Ablation only |
| UnitY-style semantic → units | Good language modeling, data reuse and fast discrete-unit synthesis | More training machinery than a cascade | Selected |
| Direct mel/spectrogram synthesis | Can preserve voice/prosody | Long continuous acoustic prediction is expensive and harder to train | Rejected initially |
| Codec-token speech LLM | Flexible and expressive | Long autoregressive acoustic sequences, high memory and compute | Rejected |

Direct and cascaded speech-translation quality can be comparable in well-resourced
settings, but direct systems are more sensitive to data scarcity. This is why the
selected system retains text supervision and a cascade teacher rather than attempting
pure speech-to-speech learning. [Cascade versus direct](https://aclanthology.org/2021.acl-long.224/),
[data-efficient speech translation](https://aclanthology.org/Q19-1020/)

For English→Hindi specifically, an IWSLT 2024 submission reported BLEU 51.0 for a
Whisper+IndicTrans2 cascade versus 19.1 for its end-to-end system. That does not prove
cascades are inherently superior; it demonstrates the cost of inadequate direct-ST
training and makes a strong teacher mandatory. [NICT IWSLT 2024](https://aclanthology.org/2024.iwslt-1.3/)

## Proposed student architecture

The following values are starting constraints, not benchmark results:

### Audio frontend

- Mono 16 kHz waveform.
- 80-bin log-mel features using 25 ms windows and 10 ms hops.
- Two strided convolution layers for 4× temporal subsampling.
- Input normalization computed per utterance or with fixed training statistics.

### Speech encoder

- 12 chunk-capable Conformer blocks.
- Model width 384; 6 attention heads; head width 64.
- Feed-forward width 1,536; convolution kernel approximately 15.
- Bounded left context and configurable right look-ahead.
- Target size: 35–55M parameters.

The encoder will first operate on complete utterances using the same layers. Streaming
will activate bounded masks and state rather than replace the architecture.

### Hindi semantic decoder

- 6 Transformer decoder blocks.
- Model width 384; 6 heads; feed-forward width 1,536.
- Hindi-focused SentencePiece vocabulary, initially 8K tokens.
- Greedy or beam-2 decoding; no general-purpose language-model prompt.
- Target size: 20–35M parameters.

Autoregressive decoding is retained initially because English→Hindi requires flexible
word reordering. A non-autoregressive semantic decoder will be evaluated only after the
quality baseline exists. DASpeech is relevant evidence that fully non-autoregressive
S2ST can substantially improve speed, but it accepts a quality trade-off.
[DASpeech](https://arxiv.org/abs/2310.07403)

### Speech-unit decoder

- UnitY2-inspired text/character-to-unit upsampling.
- 4 non-autoregressive Transformer or convolution-attention blocks, width 256.
- Explicit duration prediction.
- Unit vocabulary chosen by the target-unit tokenizer experiment, not guessed now.
- Target size: 10–20M parameters.

### Vocoder

- Compact unit-conditioned HiFi-GAN-style generator.
- Initially FP16; quantize only after intelligibility and audio quality testing.
- Target size: 10–20M parameters.

### Whole-model budget

- Target parameters: 75–130M.
- FP16 weight target: 150–260 MB.
- INT8 weight target: 75–130 MB, excluding metadata/scales.
- Peak runtime-memory target: at most 2 GB for model inference.
- Batch size: 1.
- Maximum initial utterance: 15 seconds.

Actual latency and activation memory must be measured after the executable baseline.

## State and KV-cache policy

The system has no conversational memory:

```text
start utterance → translate → finish audio → destroy all utterance state
```

This satisfies the requirement that people, rather than the machine, carry conversation
context between turns. It does **not** imply that all temporary computation should be
disabled.

During one utterance the system may hold:

- bounded Conformer attention and convolution state;
- encoded audio representations;
- a small semantic-decoder KV cache;
- incremental unit/vocoder buffers.

An autoregressive decoder without KV caching recomputes earlier tokens at every step.
The cache avoids that repeated work. For 6 layers, width 384, 128 tokens and FP16, the
self-attention K/V payload is approximately:

```text
2 (K,V) × 6 layers × 128 tokens × 384 values × 2 bytes ≈ 1.1 MiB
```

That is a good latency trade for an 8 GB device. It is created for the current utterance
and erased immediately afterward. It is not a user-history or cross-conversation cache.
[Transformer cache explanation](https://huggingface.co/docs/transformers/cache_explanation)

## Training and data strategy

The 8 GB M2 is the development and inference machine, not the sole full-training device.
The training graph uses external teachers that never ship in the final runtime.

1. Establish two baselines: SeamlessM4T English→Hindi S2S and
   Whisper→IndicTrans2→Hindi TTS.
2. Pretrain/distil the Hindi semantic decoder from English–Hindi parallel text.
3. Train English speech→Hindi semantic tokens on direct ST data and pseudo-labels.
4. Extract Hindi speech units offline using a pretrained tokenizer.
5. Train Hindi semantic/character→unit generation on Hindi text+speech.
6. Train/adapt the compact vocoder.
7. Jointly fine-tune the student on available paired examples.
8. Quantize and profile only after the FP16 model is correct.

Relevant data sources require license and contamination audits before downloading:

- Indic-TEDST includes English audio and Hindi text translations; the IWSLT 2024 setup
  describes 103.5 hours of English speech and about 37K Hindi lines.
  [Indic-TEDST](https://aclanthology.org/2024.lrec-main.790/),
  [IWSLT 2024 overview](https://aclanthology.org/2024.iwslt-1.pdf)
- Samanantar reports roughly 10.1M English–Hindi sentence pairs for semantic training.
  [Samanantar](https://arxiv.org/abs/2104.05596)
- The IIT Bombay English–Hindi corpus reports about 1.49M parallel segments.
  [IIT Bombay corpus](https://arxiv.org/abs/1710.02855)
- BhasaAnuvaad/SeamlessAlign and Indic-S2ST are candidates for additional mined,
  synthetic or parallel supervision; synthetic data must not be reported as equivalent
  to clean human-parallel speech.
  [BhasaAnuvaad](https://arxiv.org/abs/2411.04699),
  [Indic-S2ST](https://aclanthology.org/2025.ijcnlp-long.196/)

CVSS is not a primary English→Hindi training corpus: it contains 21 source languages
translated into English, not English translated into Hindi.
[CVSS](https://arxiv.org/abs/2201.03713)

## Hardware implications

### Apple M2 MacBook Air, 8 GB

Apple Silicon uses unified CPU/GPU memory, so model weights, activations and the rest of
the application compete for the same 8 GB. MLX can let CPU and GPU operate on arrays in
that shared pool without explicit copies. [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)

Development will use PyTorch/MPS for portability. Deployment experiments will use Core
ML, allowing the OS to partition supported operations across CPU, GPU and Neural Engine.
Unsupported graph fragments or frequent device transitions can erase theoretical gains,
so exported execution must be inspected and benchmarked.
[Core ML](https://developer.apple.com/documentation/coreml/),
[Core ML compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits)

Apple does not publish a directly comparable dense-INT8 TOPS figure for this M2 that can
be responsibly compared with Jetson marketing numbers. The M2 target will therefore be
sized using measured latency, memory and energy, not an invented FLOPS conversion.

### Jetson Orin Nano 8 GB

The original mode provides approximately 20 dense / 40 sparse INT8 TOPS, 10 FP16 TFLOPS
and 68 GB/s memory bandwidth. Super mode raises this to approximately 33 dense / 67
sparse INT8 TOPS, 17 FP16 TFLOPS and 102 GB/s, with a higher power envelope.
[NVIDIA comparison](https://developer.nvidia.com/blog/?p=93942),
[Orin Nano guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/)

Architecture and capacity decisions must use the dense number. Sparse TOPS assumes
compatible 2:4 structured sparsity, and TensorRT can select a dense tactic when sparse is
not faster. [TensorRT structured sparsity](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/io-formats-sparsity.html)

The ARM CPU should handle capture, VAD, resampling and orchestration. The Orin GPU should
run the frontend, encoder, decoders and vocoder. Orin Nano has no DLA cores available for
this design, so DLA offload must not be assumed. TensorRT engines must be built against
the TensorRT 10.x version supplied by the target JetPack and for Orin's `sm_87` GPU.
[JetPack 6.2](https://docs.nvidia.com/jetson/jetpack/6.2/release-notes/index.html),
[TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html)

## FLOPS/TOPS interpretation

- FLOPS measures floating-point operations; TOPS commonly reports integer operations.
- Peak figures assume favorable shapes, precision, clock, utilization and often sparsity.
- Batch-1 streaming models frequently become memory-, launch- or synchronization-bound.
- A model with fewer nominal FLOPs can be slower if it creates unsupported operators,
  device transfers, tiny kernels or long autoregressive sequences.
- Therefore the architecture is optimized against measured end-to-end latency and memory,
  not parameter count or advertised TOPS alone.

The deployment benchmark must report p50/p95 latency, time to first audio, real-time
factor, peak memory, power mode, sustained power, temperature, throttling, CPU/GPU
utilization and memory-controller utilization. On Jetson this includes `tegrastats`; on
Apple it includes Instruments/Core ML compute-plan inspection.

## Acceptance gates

1. A reproducible cascade baseline defines the attainable translation quality.
2. The FP16 student must produce intelligible Hindi before quantization.
3. The student must fit below 2 GB peak inference memory on the M2.
4. INT8 is accepted per component only when quality loss remains within a declared bound.
5. Streaming work begins from the chunk-capable encoder but is accepted only when total
   buffering is included in latency.
6. The final architecture is chosen from measurements, so layer counts may shrink after
   profiling without changing the semantic→unit design.

## Sources most influential to the decision

- [UnitY](https://arxiv.org/abs/2212.08055)
- [SeamlessM4T](https://arxiv.org/abs/2308.11596)
- [Seamless/UnitY2 and streaming](https://arxiv.org/abs/2312.05187)
- [SimulTron on-device S2ST](https://arxiv.org/abs/2406.02133)
- [Translatotron 2](https://arxiv.org/abs/2107.08661)
- [Direct speech-to-unit translation](https://arxiv.org/abs/2107.05604)
- [FastSpeech 2](https://arxiv.org/abs/2006.04558)
- [VAKTA-SETU English–Hindi cascade](https://arxiv.org/abs/2305.12518)
- [IndicTrans2](https://arxiv.org/abs/2305.16307)
