## 1. GPU layer

- [x] WGSL shaders: `embedding-gather`, `embedding-scatter-accum`, `cross-entropy-from-logits`, `cross-entropy-logits-grad`, `broadcast-fill-scalar`
- [x] register + dispatch helpers + IDs in `packages/gpu/src/core-kernels.ts`; export from `packages/gpu/src/index.ts`
- [x] `uint32` buffer helpers in `packages/gpu/src/buffer.ts`; CPU parity helpers in `cpu-reference.ts`

## 2. Tensor layer

- [x] `DType` includes **`u32`**; `createIndices` / `readIndices` / `makeIndicesTensor`
- [x] Tensor ops `embedding`, `scatterEmbeddingAccum`, `crossEntropyRowsFromLogits`, `crossEntropyLogitsGrad`, `sumFlat`, `broadcastFillFromScalar`
- [x] `reshape` restricts to **`f32`** tensors (`assertDtype`)
- [x] GPU-gated `embedding-ce.spec.ts`

## 3. Autograd layer

- [x] `embedding`, `crossEntropyRowsFromLogits`, `sumFlat`, `crossEntropyMeanFromLogits`
- [x] GPU-gated `embedding-ce.spec.ts`

## 4. OpenSpec archive

- [x] Artifact tree under `openspec/changes/archive/2026-05-10-embedding-cross-entropy/`
