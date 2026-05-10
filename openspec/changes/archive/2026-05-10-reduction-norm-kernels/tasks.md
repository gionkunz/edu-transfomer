## 1. WGSL sources and loaders

- [x] 1.1 Add `packages/gpu/src/kernels/reduce-sum.wgsl` with a documented `@compute @workgroup_size(256)` entry point. Bindings: `binding 0` uniform `(rows: u32, cols: u32, _pad0: u32, _pad1: u32)`, `binding 1` `read` storage `input: array<f32>` (length `rows * cols`, row-major), `binding 2` `read_write` storage `out: array<f32>` (length `rows`). One thread per output row; thread `i` returns early if `i >= rows`, else loops `j` over `cols` accumulating `input[i * cols + j]` into `out[i]`.
- [x] 1.2 Add `packages/gpu/src/kernels/reduce-max.wgsl` with the same binding layout as `reduce-sum.wgsl`. Initialize the per-row accumulator from the first row element (`input[i * cols]`); loop `j` from 1 to `cols` taking `max(acc, input[i * cols + j])`; write to `out[i]`. _(Originally specified as `bitcast<f32>(0xff800000u)`; Tint rejects that with "value -inf cannot be represented as 'f32'", so first-element init is used instead. Safe because the dispatch helper rejects `cols <= 0`. Spec and design updated.)_
- [x] 1.3 Add `packages/gpu/src/kernels/softmax.wgsl` with `@workgroup_size(256)`. Bindings: `binding 0` uniform `(rows, cols, _pad0, _pad1)`, `binding 1` `read` storage `input`, `binding 2` `read_write` storage `out`. One thread per row; **two-pass per row**: pass 1 finds `m = max(input[i,*])` initialized from `input[i*cols]` (same Tint constraint as 1.2); pass 2 computes `out[i,j] = exp(input[i,j] − m)` while accumulating `s`; pass 3 (or trailing loop) divides `out[i,j] /= s`. Document the stability requirement in a top-of-file comment with a pointer to `docs/project.md`'s Numerical Stability section.
- [x] 1.4 Add `packages/gpu/src/kernels/rmsnorm.wgsl` with `@workgroup_size(256)`. Bindings: `binding 0` uniform `(rows: u32, cols: u32, epsilon: f32, _pad: u32)`, `binding 1` `read` storage `input` (length `rows * cols`), `binding 2` `read` storage `gamma` (length `cols`), `binding 3` `read_write` storage `out` (length `rows * cols`). One thread per row: compute `meanSq = sum(input[i,j]² for j) / cols`, `inv = 1 / sqrt(meanSq + epsilon)`, then write `out[i,j] = input[i,j] * inv * gamma[j]`.
- [x] 1.5 Extend `packages/gpu/src/kernels/load-core-wgsl.ts` with `loadReduceSumWgsl`, `loadReduceMaxWgsl`, `loadSoftmaxWgsl`, `loadRmsNormWgsl`, mirroring the existing `loadMatMulWgsl` / `loadTransposeWgsl` style.

## 2. Registration and dispatch

- [x] 2.1 In `packages/gpu/src/core-kernels.ts`, add the four new ID constants (`CORE_KERNEL_REDUCE_SUM = 'core.reduce_sum'`, `CORE_KERNEL_REDUCE_MAX = 'core.reduce_max'`, `CORE_KERNEL_SOFTMAX = 'core.softmax'`, `CORE_KERNEL_RMSNORM = 'core.rmsnorm'`) and append them to the `CORE_KERNEL_IDS` array.
- [x] 2.2 Extend `registerCoreKernels` to compile + register the four new pipelines via the existing `make` helper. Bindings per kernel:
  - `reduce_sum`, `reduce_max`: `['uniform', 'read-only-storage', 'storage']` (3 bindings).
  - `softmax`: `['uniform', 'read-only-storage', 'storage']` (3 bindings).
  - `rmsnorm`: `['uniform', 'read-only-storage', 'read-only-storage', 'storage']` (4 bindings; gamma is a separate readonly buffer).
- [x] 2.3 Implement `dispatchReduceSum(ctx, registry, input, out, rows, cols)` in `core-kernels.ts`. Uses `makeUniform` to pack `(rows, cols, 0, 0)` into the existing 16-byte allocation; dispatches with `totalThreads = rows`. Reject `rows <= 0` or `cols <= 0` synchronously with a descriptive `Error`.
- [x] 2.4 Implement `dispatchReduceMax(ctx, registry, input, out, rows, cols)` mirroring `dispatchReduceSum`.
- [x] 2.5 Implement `dispatchSoftmax(ctx, registry, input, out, rows, cols)` mirroring `dispatchReduceSum`'s uniform packing; dispatch with `totalThreads = rows`.
- [x] 2.6 Implement `dispatchRmsNorm(ctx, registry, input, gamma, gammaLength, out, rows, cols, epsilon = 1e-6)` packing `(rows: u32, cols: u32, epsilon: f32, 0)`; dispatch with `totalThreads = rows`. JSDoc warns that `epsilon === 0` divides by zero on a zero-norm row. _(Added `gammaLength: number` parameter — `GPUBuffer` carries no element count, so the synchronous `gamma.length !== cols` validation in spec needs the length explicitly. Aligns with the spec scenario's "declared length" wording.)_
- [x] 2.7 Re-export the four new IDs and four new dispatch helpers from `packages/gpu/src/index.ts` alongside the existing `CORE_KERNEL_*` / `dispatch*` exports. **Do not** rename or remove any existing export.

## 3. CPU references

- [x] 3.1 Add `cpuReduceSum(data: Float32Array, rows: number, cols: number): Float32Array` (length `rows`) to `packages/gpu/src/cpu-reference.ts`.
- [x] 3.2 Add `cpuReduceMax(data: Float32Array, rows: number, cols: number): Float32Array` initializing each row's accumulator to `Number.NEGATIVE_INFINITY`.
- [x] 3.3 Add `cpuSoftmax(data: Float32Array, rows: number, cols: number): Float32Array` (length `rows * cols`) using the textbook two-pass per row (subtract row max, exp, divide by row sum) — same algorithm the WGSL kernel implements.
- [x] 3.4 Add `cpuRmsNorm(data: Float32Array, gamma: Float32Array, rows: number, cols: number, epsilon: number): Float32Array` (length `rows * cols`). Throws if `gamma.length !== cols`. Default `epsilon` value is **caller-supplied** (the dispatch helper carries the `1e-6` default).
- [x] 3.5 Re-export the four new CPU references from `packages/gpu/src/index.ts` alongside the existing `cpu*` exports.

## 4. Tests (Vitest)

- [x] 4.1 In the existing `packages/gpu/src/core-kernels.spec.ts` (or a sibling `reduction-norm-kernels.spec.ts` if the file would exceed ~400 lines), add a `describe.skipIf(!webgpuAvailable)` block covering:
  - `dispatchReduceSum` against `cpuReduceSum` for shapes `[3, 5]` and `[1, 16]`.
  - `dispatchReduceMax` against `cpuReduceMax` for the same shapes.
  - `dispatchSoftmax` against `cpuSoftmax` for `[2, 4]` (uniform-ish row) **and** for a deliberately-large-magnitude row such as `[1000, 1001, 999]` (`rows = 1, cols = 3`); assert every output element is finite (`Number.isFinite`) and matches the CPU reference within float tolerance. _(Landed as sibling `reduction-norm-kernels.spec.ts` to keep `core-kernels.spec.ts` focused.)_
- [x] 4.2 Add a small synchronous-error test (no GPU required) confirming that each new dispatch helper throws when `rows <= 0` or `cols <= 0` (and `dispatchRmsNorm` additionally throws if `gamma.length !== cols`). May live in the same spec file with `describe('argument validation', ...)` outside the `skipIf` block.
- [x] 4.3 Extend the existing registry-coverage check (the test that asserts every ID in `CORE_KERNEL_IDS` resolves to a pipeline after `registerCoreKernels`) so the four new IDs are exercised automatically — no per-ID duplication. _(No code change required: the existing `core-kernels.spec.ts` already iterates `CORE_KERNEL_IDS`, which now contains the four new IDs from task 2.1.)_

## 5. Verification

- [x] 5.1 Run `nx build @llm-transformer-diy/gpu`; fix any strict-TS issues introduced by the new exports. _(Clean build, no TS errors.)_
- [x] 5.2 Run root `pnpm lint` and `pnpm test`; ensure GPU-gated specs skip cleanly on environments without WebGPU and the gpu package suite still exits zero. _(Lint clean. `pnpm test`: all 9 projects passed; gpu suite is 26/26 — the prior 16 plus 5 new GPU integration tests and 5 new sync-validation tests.)_
- [x] 5.3 Sanity-check the dependency edge: `pnpm list --filter @llm-transformer-diy/gpu` shows no new runtime dependencies. `packages/tensor/package.json` is unchanged. No file under `packages/tensor/src/` is modified by this change. _(Confirmed: gpu still depends only on `webgpu@0.4.0`; `git status` shows no edits under `packages/tensor`.)_
- [x] 5.4 Grep audit: `rg 'core\.(reduce_sum|reduce_max|softmax|rmsnorm)' packages/tensor packages/autograd packages/model packages/training packages/inference apps` returns **no matches** — confirms the new IDs are not consumed by any layer above `gpu` in this change (consumers will land in their own follow-up changes). _(Confirmed: only `packages/gpu/src/core-kernels.ts` references the new IDs.)_
