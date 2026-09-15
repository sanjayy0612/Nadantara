# Build plan

This is one product developed through verified vertical slices, not separate products.

## 0 — executable foundation (complete)

Define model contracts, validate deterministic audio chunking, detect MPS/CUDA/CPU,
and exercise byte-exact audio transport.

## 1 — teacher and data truth

Audit English → Hindi datasets and licenses. Run a Seamless-family teacher on a fixed
evaluation manifest. Record quality, memory, first-output latency and real-time factor;
cache the targets needed for distillation.

## 2 — compact offline student

Integrate a compact pretrained speech encoder. Train the semantic and unit decoders
with auxiliary text supervision and initially reuse a lightweight vocoder. Exit when
English speech produces intelligible Hindi speech with no user-visible text stage.

## 3 — edge runtime

Profile on M2, apply mixed precision and INT8 compression, bound caches, export supported
graphs to Core ML, and preserve ONNX/TensorRT portability.

## 4 — streaming

Add chunk-aware encoder state, a conservative read/write policy, and incremental speech
generation without boundary artifacts. Report latency including all buffering.

## Next task

Before downloading checkpoints, select a legally usable evaluation corpus, one-way
English→Hindi teacher, compatible unit tokenizer/vocoder, and pretrained components that
fit the budgets accepted in [architecture-study.md](architecture-study.md).
