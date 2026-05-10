## Why

Sequential plan step **7** (`docs/sequential-change-proposals.md`) requires **GPU-side optimizer and gradient utilities** so a later training loop can update parameters after backward without ad hoc buffer math. The project doc specifies fused **AdamW** and **global L2 gradient clipping** (`docs/project.md`).

## What Changes

- **`packages/gpu`**: New WGSL kernel **`core.adamw_step`** (fused moment + parameter update matching the decoupled weight-decay form in `docs/project.md`), **`cpuAdamWStep`** reference, **`dispatchAdamWStep`**, registered alongside existing core kernels.
- **`packages/tensor`**: **`adamWStepInPlace`** (+ **`AdamWStepHyperparams`**) wrapping the dispatch; contiguous **`f32`** tensors only.
- **`packages/autograd`**: **`clipGradNorm(rt, reg, maxNorm)`** (async) — global norm via existing `mul`/`sumFlat`/`add` chain; per-parameter **`scale`** when `norm > maxNorm`; documents grad write paths with **`clipGradNorm`**.
- **`packages/training`**: **`AdamW`** class (lazy per-parameter **m**/**v** buffers, **`weightDecayScale`**, **`setLr`**, **`step`** clears consumed grads, **`dispose`**), default hyperconstants aligned with project doc; Vitest smoke test; depends on **`autograd`** + **`tensor`**; **`gpu`** + **`@webgpu/types`** as devDependencies for tests.

## Explicitly Out of Scope

- Cosine LR / warmup (**training loop** change), dataset batching, checkpointing optimizer state (**io** pairing later), full **`Trainer`**.

## Capabilities

### New

- **`optimizer-training-utils`**: AdamW fused step on GPU, global grad norm clipping on the parameter registry, training-package **`AdamW`** wrapper.

### Modified

- **`core-wgsl-kernels`**: inventory gains **`core.adamw_step`** (consumer-facing registry ID).

## Impact

- Primary: `packages/gpu`, `packages/tensor`, `packages/autograd`, `packages/training`.
- Downstream: model + trainer can depend on **`AdamW`** and **`clipGradNorm`** without new ML dependencies.
