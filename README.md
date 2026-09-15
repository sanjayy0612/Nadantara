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
backend, automatic MPS/CUDA/CPU selection, and the trainable compact UnitY-style PyTorch
graph. The model currently has random weights and therefore does not translate yet.

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
- [Dataset audit](docs/dataset-audit.md)
- [Short build overview](docs/build-plan.md)
- [Contributor and agent instructions](AGENTS.md)

## Repository layout

```text
dataset/          dataset schemas, manifests and provenance guidance
docs/             architecture, product and evaluation decisions
model_registry/   model identity and lineage metadata (not weights)
src/              reusable runtime and model code
tests/            deterministic unit and smoke tests
train/            experiment configurations and training entry points
```
