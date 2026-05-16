# Implementation tasks — add-16bit-dtype-support

Tasks are grouped by the three migration phases in `design.md` (inference-first → train+backward → checkpoint round-trip), prefixed by foundation work that all phases share. Each task is verifiable; commit boundaries roughly match the numbered groups.

## 1. Foundation: dtype enum, capability probe, CPU helpers

- [x] 1.1 Widen `DType` in `packages/tensor/src/types.ts` to `'f32' | 'bf16' | 'fp16' | 'u32'`; keep `Tensor.dtype` `readonly`; add a `bytesPerElement(dtype)` exported helper (re-exported from `index.ts`).
- [x] 1.2 Extend `GpuBootstrapOptions` in `packages/gpu/src/device.ts` with optional `requiredFeatures` / `optionalFeatures` (default `optionalFeatures: ['shader-f16']`); plumb both into `adapter.requestDevice`.
- [x] 1.3 Record `grantedFeatures: ReadonlySet<GPUFeatureName>` on `GpuContext`; throw a clear `Error` from `GpuContext.create` when any `requiredFeatures` entry is not granted; expose `supportsDtype(ctx, dtype)` as a free function (and document the bf16-always / fp16-conditional rules).
- [x] 1.4 Add CPU pack/unpack helpers in `packages/gpu/src/bf16-fp16-pack.ts`: `packBf16FromFloat32`, `unpackBf16ToFloat32`, `packFp16FromFloat32`, `unpackFp16ToFloat32`, plus `bytesPerElement(dtype)`; export from `packages/gpu/src/index.ts`.
- [x] 1.5 Write Vitest unit tests covering bf16 RNE rounding, bf16 representable-integer round-trip, fp16 saturation to `+Inf`, fp16 `NaN` preservation, fp16 subnormals — all CPU-only, must run without WebGPU.
- [x] 1.6 Add `createBf16StorageBuffer`, `createFp16StorageBuffer`, `writeBf16Buffer` / `writeFp16Buffer` (accepting `Uint16Array`), and `readBf16Buffer` / `readFp16Buffer` to `packages/gpu/src/buffer.ts`; export from `packages/gpu/src/index.ts`; mirror the existing f32 helper signatures. GPU storage uses **one `u32` per scalar** (low 16 bits) so parallel kernels avoid non-atomic RMW on the same word.
- [x] 1.7 Add a "round-trip with no compute" Vitest spec for both 16-bit buffer kinds, gated on WebGPU availability via the existing pattern in `packages/gpu/src/buffer.spec.ts`.

## 2. Inference-first slice — kernels

- [x] 2.1 Add a small WGSL prologue helper (in `packages/gpu/src/kernels/wgsl-prologues.ts` or equivalent) that emits the bf16 / fp16 storage aliases (`array<u32>` for bf16; `array<f16>` or `array<u32>` for fp16 depending on `shader-f16`) and the `unpack` / `pack` snippets for use inside kernels; expose it to the loader.
- [x] 2.2 Author bf16 + fp16 variants of `matmul.wgsl`, `elementwise-add.wgsl`, `elementwise-mul.wgsl`, `elementwise-scale.wgsl`, `elementwise-gelu.wgsl`, `softmax.wgsl`, `rmsnorm.wgsl`, `embedding-gather.wgsl`, `cross-entropy-from-logits.wgsl`, `reduce-sum.wgsl`, `reduce-max.wgsl`, `transpose.wgsl`, `broadcast-fill-scalar.wgsl`. Reductions and matmul inner sums accumulate in `f32` (Decision 4).
- [x] 2.3 Register the new variants under `core.<name>.bf16` and `core.<name>.fp16` in `packages/gpu/src/core-kernels.ts`; add `CORE_KERNEL_*_BF16` / `_FP16` string ID constants and matching `dispatch*Bf16` / `dispatch*Fp16` wrappers.
- [x] 2.4 Extend `packages/gpu/src/cpu-reference.ts` with dtype-aware variants of each forward CPU reference: each reference computes in `f32` and applies the dtype rounding to its result for direct comparison against GPU readback.
- [x] 2.5 Parameterize the existing kernel specs (`core-kernels.spec.ts`, `reduction-norm-kernels.spec.ts`) over `f32` / `bf16` / `fp16`; use a `dtypeTolerance(dtype)` helper and skip the fp16 cases when `supportsDtype('fp16') === false`.

## 3. Inference-first slice — tensor layer + cast op

- [x] 3.1 Add a trailing optional `dtype` parameter to `createZeros`, `createFromArray` (and the indices/u32 constructors stay unchanged); plumb the dtype into the underlying buffer choice via `createBf16StorageBuffer` / `createFp16StorageBuffer`; use {@link gpuStorageByteLength} / pack helpers where buffer bytes must match GPU layout (`bf16`/`fp16` use **4 bytes per logical scalar** in `GpuBuffer`s per §1.6).
- [x] 3.2 Teach `createFromArray` to dispatch through `packBf16FromFloat32` / `packFp16FromFloat32` when the target dtype is 16-bit; teach `toArray` to read back the 16-bit buffer and unpack to a fresh `Float32Array`.
- [x] 3.3 Update every op function in `packages/tensor/src/ops.ts` to (a) assert all tensor operands share the same dtype synchronously, (b) look up the dtype-suffixed kernel ID, (c) allocate the output buffer at the operand dtype. Keep the f32 control flow byte-identical. Backward ops (`geluBackward`, `softmaxBackward`, `rmsnorm` grads, `crossEntropyLogitsGrad`, `scatterEmbeddingAccum`) gained bf16/fp16 dispatches in §5; `adamWStep` stays f32-only until §6.
- [x] 3.4 Add `cast(rt, t, dtype)` plus `core.cast.f32.bf16`, `core.cast.bf16.f32`, `core.cast.f32.fp16`, `core.cast.fp16.f32` kernels; identity-dtype performs a non-aliasing GPU copy; `bf16`↔`fp16` chains via `f32` on the device.
- [x] 3.5 Write Vitest specs for: default dtype is `f32`; mixed-dtype ops throw before dispatch; cast round-trips `f32 → bf16 → f32` within bf16 tolerance; cast to same dtype produces a non-aliasing buffer.

## 4. Inference-first slice — model + CLI generate path

- [x] 4.1 Add an optional `dtype: DType` field to `createGptModel` (default `'f32'`); thread the dtype into every parameter allocation; preserve f32 default behavior.
- [x] 4.2 Make `collectGptParameters` / `gptParameterTensorNames` dtype-agnostic; surface each parameter's compute dtype on the returned record.
- [x] 4.3 Add `--dtype` to the `generate` command in `apps/cli/src/cmd/`; probe `supportsDtype` immediately after `createTensorRuntime`; exit non-zero with a clear error if unsupported.
- [x] 4.4 On checkpoint load, if the on-disk dtype mismatches the requested runtime dtype, cast at load time via `cast` (gate behind an `--allow-dtype-upcast` flag for now; default behavior for the f32→bf16 inference path is to cast).
- [x] 4.5 Smoke test: `packages/training/src/gpt-checkpoint-dtype.spec.ts` compares f32 vs bf16-forward logits after hydrate; CLI manual smoke: `llm generate --dtype bf16 --model …`.

## 5. Train + backward slice — kernels and autograd

- [x] 5.1 Author bf16 + fp16 variants of every backward kernel: `elementwise-gelu-backward.wgsl`, `softmax-backward.wgsl`, `rmsnorm-grad-input.wgsl`, `rmsnorm-grad-gamma.wgsl`, `cross-entropy-logits-grad.wgsl`, `embedding-scatter-accum.wgsl`; register under `core.<name>.bf16` / `.fp16`; accumulate gradients in `f32` internal arithmetic.
- [x] 5.2 Extend `packages/gpu/src/cpu-reference.ts` with dtype-aware backward references; parameterize the existing reduction / norm specs over dtype.
- [x] 5.3 Update `packages/autograd` so `Variable.grad` is allocated at the forward tensor's dtype; ensure every `Operation.backward` returns `Tensor | null` entries at the matching input dtype.
- [x] 5.4 Add `dtypeTolerance(dtype)` to the gradient-check harness (`packages/autograd/src/gradient-check.spec.ts`) and parameterize the existing checks over `f32` / `bf16` / `fp16` (skip fp16 on adapters without support).
- [x] 5.5 Run the existing autograd specs unchanged against `f32` and confirm byte-identical results vs. pre-change.

## 6. Train + backward slice — Parameter master weights + AdamW

- [x] 6.1 Add optional `masterTensor: Tensor | null` to `Parameter` in `packages/autograd/src/variable.ts` (or wherever `Parameter` is declared); add `createParameterWithMaster(rt, tensor, dtype)` helper that allocates the compute tensor at `dtype` and the master tensor at `f32`.
- [x] 6.2 Update `ParameterRegistry`: `zeroGrad` zeroes whichever tensor backs the grad (compute), `clipGradNorm` reduces over the grad dtype but accumulates the global norm in `f32`, `totalParams` counts compute-tensor elements only.
- [x] 6.3 Refactor `AdamW` (`packages/training/src/adamw.ts`) to allocate moments at `f32` regardless of parameter compute dtype, update via the f32 adamW kernel against the master tensor when present, then cast f32→compute-dtype back into `parameter.tensor` via `core.cast.f32.<bf16|fp16>`.
- [x] 6.4 Extend `packages/training/src/adamw.spec.ts` with a bf16-compute / f32-master path that asserts (a) moments stay at f32 dtype, (b) the bf16 compute tensor receives the cast-back value within bf16 tolerance, (c) pure-f32 path produces byte-identical results to the pre-change behavior.

## 7. Train + backward slice — CLI train path

- [x] 7.1 Add `--dtype` to `apps/cli/src/cmd/` train command; default `'f32'`; probe `supportsDtype` once; refuse unsupported dtype with non-zero exit.
- [x] 7.2 When `--dtype bf16` is passed, build the model with `createGptModel({ …, dtype: 'bf16' })` and request master weights on every parameter; thread the same dtype through the data pipeline (`lm-batch` stays in `i32` indices, unchanged).
- [x] 7.3 When `--dtype fp16` is passed for `train`, emit the single-line "fp16 training is unstable without loss scaling — prefer bf16" warning, then continue.
- [x] 7.4 Train `tiny` on Shakespeare for N steps (N chosen so a wall-clock measurement is < 60s on the reference adapter) in `f32` and again in `bf16`; assert the final train loss is within a documented envelope; record the measurement.
- [x] 7.5 Measure VRAM (via Dawn / OS tools — best-effort) and tokens/sec for `small` in `f32` vs `bf16`; record the result to satisfy the proposal's "measurable memory or throughput win on one reference config" success criterion.

## 8. Checkpoint round-trip slice

- [x] 8.1 Extend the `PackedCheckpointSidecar` type / parser in `packages/io` to read and persist `tensors[name].dtype`; default missing values to `'f32'`; reject any value other than `'f32' | 'bf16' | 'fp16'`.
- [x] 8.2 Update the binary stride math in the checkpoint reader / writer to use `bytesPerElement(dtype)` per tensor; preserve byte-identical output for existing f32 checkpoints.
- [x] 8.3 Update `saveGptCheckpoint` / `hydrateGptModelWeights` in `packages/training/src/checkpoint.ts` to: (a) read each parameter's compute dtype and pack at that dtype on save, (b) on load, refuse a mismatch unless `allowUpcast` is set, (c) otherwise hydrate via `createFromArray` + `cast` as needed.
- [x] 8.4 Add a Vitest spec covering: save bf16 → load bf16 (bit-identical packed bytes); save bf16 → load with `allowUpcast: 'f32'` (rounded equal); load legacy (no `dtype` field) → interpreted as f32.
- [x] 8.5 Update `llm inspect --model …` to print the per-tensor dtype alongside name and shape.

## 9. Documentation and final wiring

- [x] 9.1 Add a "Numerical stability for 16-bit dtypes" subsection to `docs/project.md`: master weights rationale, f32 accumulator policy, fp16 vs bf16 trade-offs, capability-probe behavior.
- [x] 9.2 Update the CLI section of `docs/project.md` and `README.md` to document `--dtype` on `train` and `generate`.
- [x] 9.3 Verify the full nx test run is green (`nx run-many -t test --all`) and that all WebGPU-gated 16-bit tests are skipped (not failed) when no adapter is available.
- [x] 9.4 Run `npm run build` and `npx nx run @llm-transformer-diy/cli:prune` to make sure the pruned CLI bundle still installs and `node dist/apps/cli/main.js --help` shows the new `--dtype` flag.
- [x] 9.5 Capture the measured VRAM / throughput results from task 7.5 in a short note in the change folder so the archive step has the evidence the success criterion requires.
