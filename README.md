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

## Project documentation

- [Product definition](docs/product.md)
- [Architecture decision](docs/architecture.md)
- [Model specification](docs/model-spec.md)
- [Architecture evidence](docs/architecture-study.md)
- [Implementation plan](docs/implementation-plan.md)
- [Evaluation contract](docs/evaluation.md)
- [Short build overview](docs/build-plan.md)
- [Contributor and agent instructions](AGENTS.md)
