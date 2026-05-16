# Proposal: 16-bit dtype support (bf16 / fp16) across the DIY stack

## Why

Today every layer in the stack hardcodes `f32`: `Tensor.dtype = 'f32'`, every WGSL kernel binds `array<f32>`, autograd accumulates into `f32`, training stores `f32` weights / Adam state, and checkpoints are packed `Float32`. That means peak GPU memory and matmul/elementwise bandwidth are stuck at the `f32` ceiling, which caps the largest model / longest `seqLen` we can train or even infer on a single device.

Adding an opt-in 16-bit path (**bf16 preferred for training**, **fp16 where needed**, e.g. inference on adapters that lack `shader-f16`-equivalent bfloat16 support) cuts hot-path VRAM ~2× and typically speeds up WebGPU matmul / elementwise on the same hardware. The educational goal is preserved: we write the dtype branches ourselves (bit-level pack/unpack, WGSL variants, autograd tolerances, AdamW master-weights), rather than letting a library hide the trade-offs.

## What Changes

- **Add `bf16` and `fp16` to the tensor dtype enum.** `Tensor.dtype` becomes `'f32' | 'bf16' | 'fp16' | 'u32'`; default everywhere stays `f32` and `f32`-only call sites continue to type-check.
- **Add a GPU capability probe.** `gpu-bootstrap` learns `requestFeatures(['shader-f16'])` (optional), records what was granted on `GpuContext`, and exposes a `supportsDtype(dtype)` helper. CLI / tests use this to fail loudly when a user asks for a 16-bit mode the adapter cannot honor.
- **Add 16-bit-aware buffer helpers.** New `createBf16StorageBuffer`, `createFp16StorageBuffer`, packed write helpers (CPU `Float32Array → Uint16Array` truncation for bf16 / IEEE-754 round-to-nearest-even for fp16), and matching readback helpers. `Tensor` byte length becomes dtype-driven (2 bytes for bf16/fp16, 4 bytes for f32/u32).
- **Add 16-bit variants of the core WGSL kernels** (`matmul`, `add`, `mul`, `scale`, `gelu`, `gelu_backward`, `softmax`, `rmsnorm`, `embedding`, `cross_entropy`, `reduce_sum`, `reduce_max`, `transpose`). Variants are registered as `core.<name>.bf16` / `core.<name>.fp16` alongside the existing `core.<name>` (which remains `f32`). Accumulation inside reduction kernels SHALL stay in `f32` regardless of input dtype.
- **Add a dispatch policy in `packages/tensor`.** `matmul / add / gelu / …` look up the dtype-suffixed kernel for the input dtype, require all operand dtypes to match, and allocate the output buffer at the operand dtype. Mixed-dtype calls SHALL throw synchronously before dispatch.
- **Add dtype-aware autograd tolerances.** `Variable.grad` matches the forward dtype; the finite-difference / CPU-reference test harnesses gain dtype-specific tolerances (looser for bf16 / fp16) and run for each registered dtype.
- **Add `bf16` model parameters with `f32` master weights for training (mixed precision).** `Parameter`/`ParameterRegistry` may hold a `f32` "master" tensor in addition to the `bf16` shadow used in forward/backward. AdamW updates the `f32` master, then casts back to `bf16` for the next forward pass. Inference-only models may load straight `bf16`/`fp16` parameters without a master tensor.
- **Extend the checkpoint format.** `tensors[name].dtype` becomes one of `'f32' | 'bf16' | 'fp16'`. Binary payloads are packed at the per-tensor dtype (2 or 4 bytes per element). Loaders SHALL refuse a dtype the runtime cannot support and SHALL refuse silent up/down-casting unless the caller asks for it explicitly.
- **Add a CLI dtype flag.** `llm train --dtype bf16|fp16|f32` and `llm generate --dtype bf16|fp16|f32`, default `f32`. The CLI SHALL probe the adapter at startup and fail loudly with a clear message when the requested dtype is unsupported.
- **Documentation.** `docs/project.md` "Numerical Stability" table gains rows for bf16 / fp16 (rounding error in long reductions, accumulator dtype, master-weights rationale). No new public dependencies — bf16 / fp16 conversion is implemented in-house from `Float32Array` bit twiddling.

Default behavior on every code path remains `f32`; nothing about the existing tensor / autograd / model / training / inference surface changes for callers who do not opt in.

## Capabilities

### New Capabilities

- `dtype-16bit`: Opt-in `bf16` / `fp16` dtype mode for tensors, kernels, autograd, model parameters, training (with `f32` master weights), checkpoints, and the CLI. Owns the device-capability probe, the kernel-naming convention (`core.<name>.bf16` / `core.<name>.fp16`), the bit-level CPU↔GPU conversion helpers, the mixed-precision optimizer policy, and the checkpoint dtype extension.

### Modified Capabilities

_(none — the existing bootstrap specs (`gpu-bootstrap`, `tensor-bootstrap`, `core-wgsl-kernels`, `reduction-norm-kernels`, `autograd-bootstrap`, `autograd-softmax-op`, `cli-wolf-tui-commands`) all describe **f32-only** behavior in their current scopes. We are layering a new opt-in capability on top rather than rewriting their requirements, so they remain accurate for the default path.)_

## Impact

- **Code (additive):**
  - `packages/gpu`: new buffer helpers (`createBf16StorageBuffer`, `createFp16StorageBuffer`, `writeBf16Buffer` / `writeFp16Buffer`, `readBf16Buffer` / `readFp16Buffer`); `GpuContext` records granted features and exposes `supportsDtype`; 12+ new `*.wgsl` files (bf16 / fp16 variants of the core kernels) and matching CPU references / dispatch helpers; new kernel IDs `CORE_KERNEL_*_BF16` / `CORE_KERNEL_*_FP16`.
  - `packages/tensor`: `DType` widened to `'f32' | 'bf16' | 'fp16' | 'u32'`; constructors gain a `dtype` parameter (defaulted to `'f32'`); `toArray` / `createFromArray` learn 16-bit pack/unpack paths; op functions look up the dtype-suffixed kernel and enforce equal-dtype operands.
  - `packages/autograd`: `Variable.grad` allocated at the forward tensor's dtype; gradient-check tolerances become dtype-driven; no API surface change.
  - `packages/model`: `createGptModel` gains an opt-in `dtype` (defaults to `f32`); `collectGptParameters` / `gptParameterTensorNames` unchanged in shape but each parameter exposes both its compute tensor (e.g. `bf16`) and an optional master `f32` tensor.
  - `packages/training`: `AdamW` learns the master-weight policy — moments and master weights stay `f32` even when compute parameters are `bf16`; checkpoint save/load (`saveGptCheckpoint` / `hydrateGptModelWeights`) round-trip the per-tensor dtype.
  - `packages/io`: checkpoint sidecar JSON gains `tensors[name].dtype`; binary stride math becomes dtype-aware.
  - `apps/cli`: new `--dtype` flag on `train` and `generate`; startup probe of WebGPU features with a clear fail-loud error.
- **APIs / breaking surface:** No breaking changes — every existing call site that does not pass a dtype gets `'f32'` and the existing `f32` kernels. Checkpoints written by the existing code (no `dtype` field per tensor) are interpreted as `f32` for backward compatibility.
- **Dependencies:** None added. bf16 / fp16 conversion is in-house bit-level code (matches the "own the stack" mandate in `docs/project.md`). The `shader-f16` WebGPU feature is requested optionally at adapter time and gated behind the new capability probe.
- **Tests / CI:** Each core kernel gains parameterized tests across `f32` / `bf16` / `fp16` with dtype-specific tolerances; CPU references compute in `f32` and compare against the dtype-rounded GPU output. WebGPU-gated tests continue to skip cleanly when no adapter is available.
- **Performance / memory:** No regression on the default `f32` path. On the opt-in path, success criterion is **a measurable memory or throughput win on at least one reference config** (e.g. `small`/`medium` from `docs/project.md`) on a real WebGPU adapter.
- **Risk:** WebGPU adapters that lack `shader-f16` (and have no native `bfloat16` extension) will refuse the bf16 / fp16 path — the capability probe + CLI flag combination is designed to surface that loudly rather than silently emulate.
