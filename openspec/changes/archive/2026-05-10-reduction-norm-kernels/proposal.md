## Why

`packages/gpu` ships device bootstrap and the `core.*` numeric kernels (`matmul`, `add`, `mul`, `scale`, `gelu`, `transpose`), and `packages/tensor` now wraps them as a functional op surface. The next layer up — autograd plus the transformer block — needs **stable softmax**, **row-wise reductions**, and **RMSNorm** (forward) to even define a forward pass through attention or a `TransformerBlock`. None of these exist yet. This change lands the **forward-only** WGSL kernels and dispatch helpers in `packages/gpu` so a follow-up tensor / autograd change can wrap and differentiate them, mirroring the `core-wgsl-kernels` → `tensor-bootstrap` cadence.

## What Changes

- Add WGSL compute shaders and TypeScript dispatch glue under `packages/gpu/src/kernels/` for:
  - **`reduce_sum`** — row-wise sum of a row-major 2D `f32` matrix `[rows, cols]` → `f32` vector `[rows]`. One thread per row, sequential accumulation over `cols`. Naive (no workgroup-shared parallel reduction) per the sequencing guidance to start naive and defer perf to its own change.
  - **`reduce_max`** — row-wise max of a row-major 2D `f32` matrix `[rows, cols]` → `f32` vector `[rows]`. Same dispatch shape as `reduce_sum`. Initialized with `-Infinity` per row.
  - **`softmax`** — numerically stable row-wise softmax of a 2D `f32` matrix `[rows, cols]` → `[rows, cols]`. Single fused dispatch implementing the textbook two-pass algorithm **per row** (max → exp(x − max) → sum → divide), matching `docs/project.md`'s "stable 2-pass" requirement. One thread per row in this scope.
  - **`rmsnorm`** — row-wise RMSNorm of a 2D `f32` matrix `[rows, cols]` with a learned scale `gamma` of length `[cols]` → `[rows, cols]`. Computes `y = x · gamma / sqrt(mean(x²) + ε)` per row with a configurable `epsilon` uniform (default `1e-6`, supplied by caller). One thread per row in this scope.
- Register each kernel under **deterministic IDs** (`core.reduce_sum`, `core.reduce_max`, `core.softmax`, `core.rmsnorm`) on the existing `KernelRegistry`, extending the current `registerCoreKernels` wiring (same duplicate-registration policy as `gpu-bootstrap` / `core-wgsl-kernels`).
- Add **typed dispatch helpers** matching the existing `dispatch*` style: `dispatchReduceSum`, `dispatchReduceMax`, `dispatchSoftmax`, `dispatchRmsNorm`. Each accepts `GpuContext`, `KernelRegistry`, the input/output `GPUBuffer`s, the `(rows, cols)` pair, and any kernel-specific scalars (`epsilon` for `rmsnorm`).
- Add **CPU references** in `cpu-reference.ts` (`cpuReduceSum`, `cpuReduceMax`, `cpuSoftmax`, `cpuRmsNorm`) using the **same** scalar formulas and the **same `epsilon` default** as the WGSL kernels, so tests share a single source of truth.
- Add **Vitest specs** (mirror `core-kernels.spec.ts`): GPU readbacks compared to CPU references for a few small shapes, including a deliberately-large-magnitude row to exercise softmax stability (no `NaN`, no `Infinity`). Existing `describe.skipIf(!webgpuAvailable)` gating policy is reused so the suite still exits zero on environments without WebGPU.
- **Explicitly omit**: backward kernels for any of these ops (autograd lives in step 5), `causal_mask` (deferred per sequencing-doc guidance to keep diffs focused), per-axis variants other than row-wise on rank-2, dtypes other than `f32`, fused workgroup-shared parallel reductions / tiled softmax (defer to a perf pass), `embedding`, `cross_entropy`, optimizer kernels, broadcasting, **and any edits to `packages/tensor`** (a follow-up change wraps these new dispatch helpers as tensor ops).

## Capabilities

### New Capabilities

- `reduction-norm-kernels`: Requirements for forward `reduce_sum`, `reduce_max`, `softmax` (stable two-pass), and `rmsnorm` as row-wise WGSL compute kernels in `packages/gpu`, including registry IDs (`core.reduce_sum`, `core.reduce_max`, `core.softmax`, `core.rmsnorm`), typed dispatch helpers, CPU-reference validation for small inputs, and numerical-stability coverage for softmax on large-magnitude rows.

### Modified Capabilities

- _(none — `gpu-bootstrap`, `core-wgsl-kernels`, and `tensor-bootstrap` behavior is unchanged. This change extends the kernel set and its registration but does not alter any existing public surface.)_

## Impact

- **Primary**: `packages/gpu` — new `.wgsl` files under `src/kernels/` (`reduce-sum.wgsl`, `reduce-max.wgsl`, `softmax.wgsl`, `rmsnorm.wgsl`), additions to `core-kernels.ts` (new IDs, `registerCoreKernels` extension, four new dispatch helpers), additions to `cpu-reference.ts` (four CPU references), additions to `kernels/load-core-wgsl.ts` (four loaders), and Vitest coverage in `core-kernels.spec.ts` (or a sibling spec if it keeps reviews focused).
- **Dependencies**: No new npm dependencies. Stays inside the `docs/project.md` allowlist.
- **Downstream**: `packages/tensor` can wrap these kernel IDs as ops in a separate change; `packages/autograd` can later define backward functions against the forward kernels added here. No edits required in `tensor` / `autograd` / `model` for this change.
