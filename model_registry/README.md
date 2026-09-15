# Model registry

This registry tracks model identity and lineage without committing checkpoint weights.
Each entry identifies its architecture, training data, license, precision, content hash
and intended role.

Checkpoint files belong in `model_registry/checkpoints/` and are ignored. A registry
entry is not evidence that a model is accurate or fast; benchmark artifacts must support
those claims separately.
