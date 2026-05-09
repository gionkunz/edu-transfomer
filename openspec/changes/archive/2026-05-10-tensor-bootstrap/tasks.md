## 1. Package layout and types

- [x] 1.1 Add new source modules under `packages/tensor/src/`: `types.ts` (`DType = 'f32'`, `Device = 'gpu'` literal-union types; `Tensor` and `TensorRuntime` `readonly` interfaces), `shape.ts` (pure helpers), `tensor.ts` (constructors + lifecycle), `ops.ts` (op functions). **No class declarations** anywhere in the package.
- [x] 1.2 Replace the stub-only `packages/tensor/src/index.ts` with real exports: types (`Tensor`, `TensorRuntime`, `DType`, `Device`), shape helpers (`computeContiguousStrides`, `productOfShape`, `isContiguous`, `sameShape`, `assertRank`, `assertSameShape`), constructors / lifecycle (`createZeros`, `createFromArray`, `toArray`, `destroy`), and ops (`matmul`, `add`, `mul`, `scale`, `gelu`, `transpose`, `reshape`). Drop `TENSOR_LAYER_MARKER` / `__tensorDependsOnGpu` only if nothing in the workspace still imports them. _(Kept `TENSOR_LAYER_MARKER` because `@llm-transformer-diy/autograd` stub still imports it; dropped `__tensorDependsOnGpu` — unused.)_
- [x] 1.3 Confirm `packages/tensor/package.json` keeps `@llm-transformer-diy/gpu` as the only **runtime** workspace dependency (no upward edges, no new runtime packages). _(Added `@webgpu/types` as `devDependency` — types-only, no runtime — so the global `GPUBuffer` resolves under type-aware ESLint rules; loaded via `/// <reference types="@webgpu/types" />` at the top of `types.ts`. Proposal and design amended to reflect this.)_

## 2. Pure shape helpers (no GPU)

- [x] 2.1 Implement `computeContiguousStrides(shape: readonly number[]): readonly number[]` in row-major order, with a unit-test-friendly contract (e.g. `computeContiguousStrides([2,3,4]) === [12,4,1]`).
- [x] 2.2 Implement `productOfShape(shape: readonly number[]): number` (returns `1` for `[]`, throws on negative dims).
- [x] 2.3 Implement predicates: `isContiguous(t: Tensor): boolean` (compares `t.strides` to `computeContiguousStrides(t.shape)`) and `sameShape(a: Tensor, b: Tensor): boolean`.
- [x] 2.4 Implement assertion helpers: `assertRank(t: Tensor, n: number)` and `assertSameShape(a: Tensor, b: Tensor)`. Each throws a descriptive `Error` and **never** mutates inputs.

## 3. Runtime, constructors, and lifecycle

- [x] 3.1 Define `TensorRuntime` as `readonly { ctx: GpuContext; registry: KernelRegistry }`. **Do not** add a constructor function — callers compose it inline (`{ ctx, registry }`). **Do not** add `setDefaultTensorRuntime` / `getDefaultTensorRuntime` or any module-level mutable state.
- [x] 3.2 Implement `createZeros(rt: TensorRuntime, shape: readonly number[]): Tensor` — allocates a buffer via `createFloat32StorageBuffer`, zero-fills via `writeFloat32Buffer` of an `n`-sized `Float32Array`, returns a `Tensor` literal `{ shape, strides: computeContiguousStrides(shape), dtype: 'f32', device: 'gpu', buffer, length: n }`.
- [x] 3.3 Implement `createFromArray(rt: TensorRuntime, data: Float32Array, shape: readonly number[]): Tensor` — validates `data.length === productOfShape(shape)`, allocates + uploads via `writeFloat32Buffer`, returns the `Tensor` literal.
- [x] 3.4 Implement `toArray(rt: TensorRuntime, t: Tensor): Promise<Float32Array>` using `readFloat32Buffer` from `@llm-transformer-diy/gpu` (`byteLength = t.length * 4` via `float32ByteLength`).
- [x] 3.5 Implement `destroy(t: Tensor): void` calling `t.buffer.destroy()`. **No** internal alive-flag tracking; document in JSDoc that subsequent `toArray` / op calls will surface as WebGPU validation errors.

## 4. Op functions

- [x] 4.1 `matmul(rt, a, b): Tensor` — `assertRank(a, 2)`, `assertRank(b, 2)`, assert `a.shape[1] === b.shape[0]`; allocate output `[M, N]` (without zero-fill since `core.matmul` writes every cell); route to `dispatchMatMul(ctx, registry, a.buffer, b.buffer, out, M, N, K)`; return new `Tensor` literal.
- [x] 4.2 `add(rt, a, b): Tensor` and `mul(rt, a, b): Tensor` — `assertSameShape(a, b)`; allocate output of identical shape; route to `dispatchAdd` / `dispatchMul` with `elementCount = a.length`.
- [x] 4.3 `scale(rt, a, k: number): Tensor` — allocate output of identical shape; route to `dispatchScale`.
- [x] 4.4 `gelu(rt, a): Tensor` — allocate output of identical shape; route to `dispatchGelu`.
- [x] 4.5 `transpose(rt, a): Tensor` — `assertRank(a, 2)`; output shape `[a.shape[1], a.shape[0]]`; route to `dispatchTranspose`.
- [x] 4.6 `reshape(a: Tensor, newShape: readonly number[]): Tensor` — **no `rt` parameter**. Assert `productOfShape(newShape) === a.length`, assert `isContiguous(a)` (else throw `'reshape: non-contiguous strides not supported in tensor-bootstrap'`), and return a new `Tensor` literal that **shares** `a.buffer` with `shape: newShape` and `strides: computeContiguousStrides(newShape)`. Document buffer-sharing in JSDoc so callers know `destroy(a)` invalidates the reshape result and vice versa.
- [x] 4.7 Audit `ops.ts` to confirm **no `this`**, **no `class`**, **no module-level mutable variables**; ESLint-friendly `function` declarations or `const` arrow functions only.

## 5. Tests (Vitest)

- [x] 5.1 Add `packages/tensor/src/shape.spec.ts` with pure unit tests for `computeContiguousStrides`, `productOfShape`, `isContiguous`, `sameShape`, `assertRank`, `assertSameShape` — no GPU, no `await`. _(16 tests.)_
- [x] 5.2 Add `packages/tensor/src/reshape.spec.ts` with pure unit tests covering: shape product match, element-count mismatch throws, contiguous tensor reshape returns a new record sharing the same `buffer` reference. Buffer sharing can be verified by reference equality on a fake `Tensor` literal — no GPU needed for this spec. _(5 tests.)_
- [x] 5.3 Add `packages/tensor/src/tensor.spec.ts` (GPU-gated via `describe.skipIf(!webgpuAvailable)`, mirroring `packages/gpu/src/core-kernels.spec.ts`):
  - `createFromArray` + `toArray` round-trip equality;
  - `createZeros` produces a zero-filled buffer of the requested shape;
  - `matmul`, `add`, `mul`, `scale`, `gelu`, `transpose` against the matching `cpu*` reference from `@llm-transformer-diy/gpu` for one small shape each;
  - `reshape` followed by `toArray` preserves data on round-trip readback and shares the underlying buffer;
  - shape-mismatch / rank-mismatch errors are thrown synchronously **before** any GPU dispatch (`matmul` inner-dim mismatch, `add` shape mismatch, `transpose` rank ≠ 2). _(10 tests.)_
- [x] 5.4 Build the test `TensorRuntime` once per `describe` block via `{ ctx: await GpuContext.create(), registry }` and tear down with `destroy(t)` for each tensor in `afterEach`. **Do not** add a global default-runtime helper to make tests shorter.

## 6. Verification

- [x] 6.1 Run `nx build @llm-transformer-diy/tensor`; fix any strict-TS issues introduced by the new public surface. _(Clean build; no TS errors.)_
- [x] 6.2 Run root `pnpm lint` and `pnpm test`; ensure GPU-gated specs skip cleanly on environments without WebGPU and the tensor package suite still exits zero. _(Lint clean. `pnpm test` succeeded for all 9 projects; tensor: 31/31 — 16 shape unit tests, 5 reshape unit tests, 10 GPU-backed integration tests.)_
- [x] 6.3 Sanity-check the dependency edge: `pnpm list --filter @llm-transformer-diy/tensor` shows `@llm-transformer-diy/gpu` (workspace link) as the only runtime dep and `@webgpu/types` as the only devDep; no other external runtime packages appear in `packages/tensor/package.json`.
- [x] 6.4 Grep audit: scanning `packages/tensor/src` for `^class\s` / `\bthis\.` returns **no production matches**. This is the structural guard that the package stayed functional.
