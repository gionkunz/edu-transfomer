# Tasks — training-loop-bootstrap

- [x] Cosine LR + warmup (`scheduler.ts`, unit tests)
- [x] `createLmBatch` for next-token LM
- [x] `Trainer` + `registerModelParameters` + gradient-clip gating
- [x] `saveGptCheckpoint` / `loadGptCheckpointTensors` using `io` packed format
- [x] `gptParameterTensorNames` + model package export
- [x] Training package public exports (`index.ts`)
- [x] GPU Vitest: overfit single batch; checkpoint tensor key round-trip
- [x] OpenSpec archive for this change
