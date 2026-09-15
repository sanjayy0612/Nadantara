# Product definition

## Goal

SeamTranslation converts a bounded English speech utterance into intelligible Hindi
speech locally on constrained hardware.

```text
English speaker → microphone → local translator → speaker → Hindi listener
```

The user does not interact with transcripts, prompts, model selection, or cloud APIs.

## Initial user experience

1. The user starts or triggers recording.
2. The system captures one English utterance.
3. It detects the utterance boundary.
4. It produces a Hindi translation and begins playback.
5. It deletes all temporary state before accepting the next utterance.

This is turn-based translation initially, not full-duplex simultaneous interpretation.
The model is designed so streaming can later reduce perceived delay without replacing
the core architecture.

## Fixed scope

- Direction: English speech → Hindi speech.
- Deployment: offline, batch 1.
- Development device: Apple M2 MacBook Air with 8 GB unified memory.
- Edge target: Jetson Orin Nano with 8 GB shared memory.
- Input: mono, signed 16-bit PCM, 16 kHz.
- Initial utterance limit: 15 seconds.
- Internal Hindi semantic tokens are permitted but not exposed.

## Initial non-goals

- Hindi → English translation.
- General multilingual translation.
- Voice cloning or guaranteed source-voice preservation.
- Long-lived conversational context.
- Full-duplex speech where both people talk simultaneously.
- Cloud-hosted inference as a product requirement.
- Matching every task supported by SeamlessM4T.

## Success criteria

The project succeeds when the compact model:

- produces intelligible Hindi that preserves the source meaning;
- runs without a network connection;
- fits below 2 GB peak model-inference memory on the M2 target;
- sustains real-time factor below 1.0 after streaming is enabled;
- reports time-to-first-audio and translation quality on a fixed evaluation manifest;
- resets all utterance state reliably;
- exports through at least Core ML, with a maintained ONNX/TensorRT-compatible graph.

These are targets until measured. Results must identify device, OS/runtime, precision,
power mode, input duration, warm-up procedure, and evaluation commit.
