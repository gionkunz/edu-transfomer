## Why

Sequential step 6 needs **embedding forward** (**gather**) and **softmax cross-entropy** (forward + logits backward wired to softmax probabilities). Without these primitives, downstream training cannot expose a canonical loss rooted in the GPU stack.

## What Changes

- `packages/gpu`: WGSL kernels + registry IDs (`core.embedding_gather`, `core.embedding_scatter_accum`, `core.cross_entropy_from_logits`, `core.cross_entropy_logits_grad`, `core.broadcast_fill_scalar`), dispatch helpers, u32 storage buffer utilities, CPU references (`cpuEmbedding*`, `cpuCrossEntropy*`).
- `packages/tensor`: `dtype: 'u32'` index tensors (`createIndices`), f32 embedding / CE ops, `sumFlat` + `broadcastFillFromScalar`.
- `packages/autograd`: `embedding`, `crossEntropyRowsFromLogits`, `sumFlat`, `crossEntropyMeanFromLogits` graph ops.
- GPU-gated Vitest coverage in tensor + autograd.

## Capabilities

### New Capabilities

- `embedding-cross-entropy`: WGSL kernels, tensor/autograd wrappers, deterministic kernel IDs above, numerical checks vs CPU reference.

### Modified Capabilities

- _(none — additive surface only.)_

## Impact

- Layering preserved: `gpu` → `tensor` → `autograd`.
- No new npm runtime dependencies beyond `docs/project.md` policy.
