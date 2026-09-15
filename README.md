# SeamTranslation

Compact, Seamless-inspired English → Hindi speech-to-speech translation for
low-power ARM devices. Apple Silicon is the development target; the model must remain
exportable to Core ML and TensorRT.

```text
waveform → streaming speech encoder → translation core
         → speech-unit decoder → lightweight vocoder → waveform
```

Text can provide auxiliary training supervision, but the product interface is audio
in and audio out.

## Current status

The repository contains the executable system skeleton: validated audio configuration,
deterministic PCM chunking, contracts for every model boundary, an identity audio
backend, and automatic MPS/CUDA/CPU selection. The identity backend deliberately does
no translation; it validates transport without pretending that the model exists.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
seamtranslation inspect
```

Validate the audio path with a mono, 16-bit, 16 kHz PCM WAV:

```bash
seamtranslation passthrough input.wav output.wav
```

See the [architecture decision](docs/architecture.md), its
[evidence-backed study](docs/architecture-study.md), and the
[build plan](docs/build-plan.md).
