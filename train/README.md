# Training workspace

Training entry points and versioned experiment configurations live here. Large outputs
are written to `train/runs/`, which is ignored.

The training code must:

- read a validated manifest rather than scan arbitrary directories;
- record the Git commit, config, model-registry IDs and manifest hash;
- set random seeds and report nondeterministic operations;
- separate teacher-target extraction from student training;
- save resumable checkpoints outside Git;
- never label synthetic translations or speech as human data.

`train/configs/tiny-smoke.json` is deliberately too small for useful translation. It
exists to exercise the training graph on CPU/MPS without exhausting the 8 GB Mac.
