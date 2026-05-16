# Design: 16-bit dtype support (bf16 / fp16)

## Context

Every layer below the CLI assumes `f32`:

- `packages/gpu` ships `createFloat32StorageBuffer`, `writeFloat32Buffer`, `readFloat32Buffer`, and 16-plus WGSL kernels (`matmul.wgsl`, `elementwise-*.wgsl`, `softmax.wgsl`, `rmsnorm*.wgsl`, `embedding-*.wgsl`, `cross-entropy-*.wgsl`, `reduce-*.wgsl`, `transpose.wgsl`, `adamw-step.wgsl`) that bind `array<f32>` everywhere.
- `packages/tensor` exposes `DType = 'f32' | 'u32'`, `Device = 'gpu'`, and a single hard-coded `Tensor.dtype = 'f32'` in every constructor / op.
- `packages/autograd` allocates `Variable.grad` via `createZeros` (implicit `f32`) and the gradient-check tests compare to a fixed `f32` finite-difference tolerance.
- `packages/training/adamw.ts` and `packages/training/checkpoint.ts` move `Float32Array` payloads in and out of GPU; the checkpoint sidecar already records `"dtype": "f32"` per tensor but loaders ignore it.
- `apps/cli/src/runtime.ts` calls `GpuContext.create()` with no feature requests, so even adapters that _could_ honor `shader-f16` never see it requested.

The user-facing motivation (see `proposal.md`): we want to cut hot-path VRAM ~2× and usually gain WebGPU matmul/elementwise throughput, while preserving the educational mandate of `docs/project.md` — we keep writing the dtype branches ourselves rather than delegating to a library. Default everywhere stays `f32`; the 16-bit path is opt-in and probed at startup.

## Goals / Non-Goals

**Goals:**

- Introduce `bf16` and `fp16` as first-class storage dtypes for `Tensor`, end-to-end (allocation, upload, op dispatch, readback, autograd, training, checkpoint, CLI).
- Keep the `f32` default code path **byte-identical** in behavior: no existing test changes its expected numerics, no constructor signature breaks for callers who don't pass a dtype.
- Probe GPU capabilities once at `GpuContext.create` and expose `supportsDtype(d)`; fail loudly when the user asks for an unsupported dtype.
- Mixed-precision training: `bf16` compute parameters with `f32` master weights + `f32` Adam moments, all updates kernel-fused.
- Per-tensor dtype in checkpoints, with backward-compatible reads (missing `dtype` ⇒ `f32`).
- Provide measurable VRAM or throughput evidence on at least one reference config (e.g. `small` from `docs/project.md`) on a real adapter.
- Stay within the "allowed dependencies" list from `docs/project.md`: no new packages.

**Non-Goals:**

- TF32, int8, fp8, fp4, or any quantization scheme.
- Mixed-dtype ops in a single kernel call (e.g. `bf16 @ fp16`). Operands must agree.
- Tensor Cores / `subgroup`-specific matmul tuning. We keep our naive WGSL matmul shape (`@workgroup_size(256)`, single inner loop) and only swap the storage / accumulator types — perf optimization beyond the natural bandwidth win is out of scope for this change.
- Dropping the `f32` kernels; both dtypes coexist.
- A separate "inference-only" runtime split. Inference and training share the same dtype infrastructure; mixed-precision master weights are opt-in inside `AdamW`.
- Backward / autograd of `cast(t, dtype)` as a learnable op. Casts are at the parameter boundary, treated as a one-way conversion.

## Decisions

### Decision 1: `bf16` is the default for training; `fp16` is the fallback for inference / unsupported adapters

**Choice:** Treat `bf16` as the recommended training dtype, `fp16` as the recommended inference dtype on adapters that lack bf16 emulation, and `f32` as the universal fallback. WebGPU itself has no `bfloat16` feature today, so on the GPU we represent bf16 as `u16` storage and convert to `f32` for compute inside the shader (mantissa truncation/expansion is two shifts). When the adapter grants `shader-f16` we can use `f16` natively for fp16 storage _and_ compute; otherwise we represent fp16 as `u16` storage and convert in-shader as well. (For our naive kernels the in-shader expansion is free relative to the memory-bandwidth saving.)

**Alternatives considered:**

- _fp16 everywhere_: narrower dynamic range than bf16 (5-bit exponent vs 8-bit) historically destabilizes attention scores and softmax. We'd be back-porting loss-scaling tricks. Rejected — bf16 is strictly easier to teach and stabilize.
- _bf16 with `f16` storage + emulated bf16 compute via `shader-f16`_: `f16` doesn't have bf16's exponent range; you'd lose the very property we want. Rejected.
- _Require `shader-f16` for any 16-bit path_: would lock out adapters without that feature even for bf16, which we can implement purely via `u16` storage. Rejected.

**Rationale:** This matches industry practice (PyTorch / JAX both default to bf16 for training when available) and lets the educational story stay clean: bf16 = "same exponent as f32, fewer mantissa bits"; fp16 = "5-bit exponent, more mantissa bits, narrower range — needs care".

### Decision 2: Dtype-suffixed kernel IDs (`core.matmul.bf16`, `core.matmul.fp16`) — not generic dispatch

**Choice:** Each existing core kernel gets two siblings registered under `core.<name>.bf16` and `core.<name>.fp16`. The existing `core.<name>` retains exact `f32` semantics. `packages/tensor` op functions look up the kernel ID via a small `kernelIdFor(name, dtype)` helper. Dispatchers in `packages/gpu` also gain dtype-specific `dispatchMatMulBf16` / `dispatchMatMulFp16` wrappers so the public surface mirrors `dispatchMatMul`.

**Alternatives considered:**

- _One kernel per op that branches on a pipeline constant_: WGSL does not support runtime dtype polymorphism, and pipeline-constant `f16` requires `shader-f16` — defeats the bf16-without-`shader-f16` plan.
- _Implicit dispatch from `Tensor.dtype`_: still requires a per-dtype WGSL source. The suffix scheme is more honest about that fact and stays grep-able in the kernel registry.

**Rationale:** Each WGSL file remains short, fully readable, and shows the dtype mechanics inline — matches `docs/project.md` "implementation prioritizes clarity, educational value". The registry stays a flat string→pipeline map.

### Decision 3: bf16 storage is `u16` with in-shader cast; fp16 storage prefers `f16` when `shader-f16` is available

**Choice:**

- **bf16 storage**: `Uint16Array` on the CPU, `array<u32>` (packed 2-per-word) in WGSL — index math is `i/2` + `i%2` shift. CPU pack is "drop low 16 mantissa bits with round-to-nearest-even"; CPU unpack is "left-shift 16, reinterpret as `f32`". In-shader unpack: `bitcast<f32>(u16_value << 16u)`; pack: `u32((bitcast<u32>(f32_value) + 0x00008000u) >> 16u)` (RNE bias).
- **fp16 storage**: `Uint16Array` on the CPU; in WGSL we prefer `array<f16>` (requires `enable f16;` and the `shader-f16` feature) when granted, fall back to `array<u32>` + manual `unpack2x16float` (which is a WGSL builtin available without `shader-f16`) when not.

**Alternatives considered:**

- _Always emulate fp16 with `u16` even when `shader-f16` is available_: forfeits ~2× compute throughput on adapters that support `shader-f16`. Rejected.
- _Use `vec2<u32>` packing layouts everywhere for symmetry_: harder to read and gives nothing back. Rejected.

**Rationale:** This is the minimum-machinery path that (a) honors the "GPU required, fail loudly" mindset (we ask for `shader-f16`, document what we did or didn't get), and (b) keeps the bf16 path universal because every WebGPU adapter supports `u16`-packed storage today.

### Decision 4: Accumulators stay in `f32`, regardless of input dtype

**Choice:** Every reduction (matmul inner sum, `reduce_sum`, `softmax` `max`+`sum`+`exp`, `rmsnorm` mean-square, `cross_entropy` log-sum-exp) accumulates into an `f32` local variable and writes the dtype-narrowed result at the very end. The matmul inner loop unpacks `bf16`/`fp16` operands to `f32`, multiplies-accumulates in `f32`, then narrows on store.

**Alternatives considered:**

- _Accumulate in the input dtype to "stay true" to the storage format_: catastrophic for long matmul reductions in bf16 (8 mantissa bits) and well-known failure mode for fp16. Rejected.
- _Accumulate in `f16` for fp16 paths to squeeze extra perf_: not worth the numerical pain at our scale. Rejected.

**Rationale:** This is the dominant industry pattern and the only one that lets bf16 training match `f32` training closely on our toy configs (`tiny`/`small`/`medium`).

### Decision 5: Mixed-precision training uses an explicit `f32` master tensor on `Parameter`

**Choice:** `Parameter` keeps `tensor: Tensor` (compute, e.g. `bf16`) and gains optional `masterTensor: Tensor | null` (always `f32` when set). `AdamW` operates on `masterTensor` (or `tensor` when no master is set), updates the moments in `f32`, then writes a narrowed copy back into `tensor` for the next forward pass. The cast back is its own fused WGSL kernel (`core.cast_f32_to_bf16` / `core.cast_f32_to_fp16`).

**Alternatives considered:**

- _Hold both copies as separate parameters in the registry_: doubles `parameters()` length and breaks `clipGradNorm`. Rejected.
- _Keep weights in bf16 and just upcast the gradient before AdamW_: well-known to drift over many steps because the weight delta `lr · m̂ / (√v̂ + ε)` quickly falls below bf16's resolution. Rejected.
- _Use loss-scaling instead of master weights (NVIDIA-style fp16 mixed precision)_: needed for fp16 _training_; we're explicitly recommending bf16 for training, which doesn't need loss scaling. We keep loss-scaling out of scope and document fp16-training as "use master weights too".

**Rationale:** Master weights are the textbook mixed-precision recipe. Doing it ourselves in TypeScript + WGSL is squarely in scope for the educational goal of the project.

### Decision 6: Capability probe is part of `GpuContext.create`, not a separate call

**Choice:** `GpuBootstrapOptions` gains an optional `requiredFeatures` / `optionalFeatures` pair (defaulted to `optionalFeatures: ['shader-f16']`). `GpuContext` records `grantedFeatures: ReadonlySet<GPUFeatureName>` and exposes `supportsDtype(d: DType): boolean`. `bf16` is always supported (we implement it with `u16` + manual cast), `fp16` is supported when either `shader-f16` is granted **or** we accept the `unpack2x16float` fallback path (we do, for inference). The CLI uses this same `supportsDtype` to decide whether to start in 16-bit mode.

**Alternatives considered:**

- _Always require `shader-f16`_: locks out current Dawn-on-CPU adapters that don't grant it. Rejected.
- _Late probe per-op_: the dtype decision affects buffer allocations from step zero (model creation), so a late probe would leave the runtime in a half-allocated state on first failure. Rejected.

**Rationale:** Single decision point; pairs cleanly with the existing `isLikelyCpuSoftwareAdapter` field.

### Decision 7: Checkpoint format extends in place, defaults to `f32` for backward compatibility

**Choice:** The sidecar's `tensors[name].dtype` field becomes one of `'f32' | 'bf16' | 'fp16'`. Binary payloads remain a flat concatenation but with per-tensor stride `productOfShape(shape) * bytesPerElement(dtype)`. When `dtype` is absent we read `f32` (existing checkpoints keep working). When the runtime cannot satisfy the requested dtype, the loader throws — we never silently up/down-cast unless the caller passes `--allow-dtype-upcast`.

**Alternatives considered:**

- _Bump a major checkpoint version_: heavy-handed for an additive field. Rejected.
- _Always store as `f32` and cast at load_: gives back the 2× disk savings we want from training in bf16. Rejected.

**Rationale:** Minimal blast radius on `packages/io` (one helper, one schema field), keeps existing checkpoints loadable, lets bf16 / fp16 training write half-size checkpoints by default.

### Decision 8: Bit-level pack/unpack lives in `packages/gpu`, not `tensor`

**Choice:** `packBf16FromFloat32(src, dst)`, `unpackBf16ToFloat32(src, dst)`, `packFp16FromFloat32(src, dst)`, `unpackFp16ToFloat32(src, dst)` are pure CPU helpers exported from `@llm-transformer-diy/gpu` alongside the buffer creators. `packages/tensor`'s `createFromArray` / `toArray` call them based on the requested dtype.

**Alternatives considered:**

- _Put them in `packages/utils`_: forces a new dependency edge in `packages/gpu`. Rejected.
- _Put them in `packages/tensor`_: makes the pack/unpack inseparable from the kernel registry, which already lives in `gpu`. Rejected.

**Rationale:** The dtype byte layout is a GPU-storage concern, and `packages/gpu` is the layer that already owns the byte-level helpers (`writeFloat32Buffer`, etc.).

### Decision 9: Default `Tensor.dtype` stays `'f32'`; every constructor adds an optional trailing dtype

**Choice:** `createZeros(rt, shape, dtype = 'f32')`, `createFromArray(rt, data, shape, dtype = 'f32')`, `createIndices(rt, ids, dtype = 'u32')` (no change), `makeIndicesTensor` unchanged. Ops infer the output dtype from the (single shared) input dtype, throw on mismatch.

**Alternatives considered:**

- _Tag dtype via the `data` array kind_ (`Float32Array` ⇒ `f32`, `Uint16Array` ⇒ infer from context): ambiguous (bf16 vs fp16 both use `Uint16Array`). Rejected.
- _Promote dtype to a `TensorRuntime` field_: locks the entire runtime to one dtype, defeats inference-in-fp16 + load-from-bf16-checkpoint workflows. Rejected.

**Rationale:** Trailing optional parameter is the least intrusive way to extend the surface; matches the JS / TS conventions already used in the repo.

## Risks / Trade-offs

- **[Risk] WebGPU adapters that don't grant `shader-f16` _and_ don't honor the `u16` storage path correctly.** This is theoretical — `u16` packed in `u32` storage is mandated by the spec — but Dawn-CPU backends have historically been buggy.
  **→ Mitigation:** Capability probe on startup, plus per-kernel "agreement with CPU reference within dtype tolerance" tests. If the probe says yes but tests disagree, we fail loudly and document a known-bad adapter list.

- **[Risk] bf16 numerics drift over long training runs even with master weights** (e.g. very small `lr` falls below bf16 resolution when written back as the compute parameter).
  **→ Mitigation:** Master weights are `f32`; the only bf16 quantity is the _value used in the next forward pass_. Mantissa loss across one step is bounded by `2^-8 · max(|θ|)`, which is well above any reasonable LR for our reference configs. Document the bound and add a regression test that trains `tiny` for N steps in bf16 vs `f32` and asserts loss curves stay within a documented envelope.

- **[Risk] fp16 overflow in softmax / attention scores.** `fp16` max is ~65 504; softmax `exp(score)` can easily exceed that.
  **→ Mitigation:** Reductions accumulate in `f32` (Decision 4); the softmax 2-pass `max` subtraction already runs in `f32`; we narrow only the final probabilities. Add a test for adversarial logits in fp16 mode that asserts no NaN/Inf at the output.

- **[Risk] Kernel duplication doubles WGSL source surface** (~12 files × 3 dtypes = ~36 files).
  **→ Mitigation:** Share the algorithmic skeleton in a small WGSL "include" stitched at load time in `packages/gpu/src/kernels/load-core-wgsl.ts` (or via a tiny string-template helper). The shared body is short — the per-dtype prologue (`enable f16;`, alias `T = bf16Word`, `unpack`/`pack` functions) is what changes. Tracked as a sub-task; if the include scheme proves too clever, we fall back to plain duplication, which is still readable.

- **[Risk] Checkpoint backward compatibility creep.** Once we admit `dtype` per tensor, the next step is `shape encoding`, `compression`, etc.
  **→ Mitigation:** Lock the schema additions to exactly `'f32' | 'bf16' | 'fp16'` for this change. Anything else gets a separate proposal.

- **[Trade-off] No fp16 loss scaling.** A user who picks `--dtype fp16` for training will likely diverge on anything but `tiny`.
  **→ Mitigation:** Documented as "fp16 is intended for inference; for training, prefer bf16 or expect to add loss scaling". The CLI emits a one-line warning when `train --dtype fp16` is requested.

- **[Trade-off] No mixed-dtype ops (e.g. `bf16 @ f32 → bf16`).** Simpler dispatch and clearer mental model at the cost of explicit `cast` ops in graph code.
  **→ Mitigation:** Provide an explicit `cast(rt, t, dtype)` op + WGSL kernels so that callers can be precise. `cast` is non-differentiable for this change (forward-only).

## Migration Plan

Roll out in three phases under one OpenSpec change so the spec deltas land together but tasks can ship sequentially:

1. **Inference-first slice.**
   - Land the dtype enum widening, `GpuContext` capability probe, CPU pack/unpack helpers, `Tensor` constructor dtype parameter, `cast` op, bf16/fp16 variants of forward-only kernels (`matmul`, elementwise, `softmax`, `rmsnorm`, `embedding`), and checkpoint dtype field.
   - CLI: `llm generate --dtype bf16|fp16` works end-to-end against an `f32` checkpoint via a one-time cast at load.
   - Success: existing `f32` tests stay green; new dtype tests pass on a real adapter; generate produces equivalent samples within dtype tolerance.

2. **Train + backward slice.**
   - Add backward kernels for the bf16/fp16 variants (`gelu_backward`, `softmax_backward`, `rmsnorm_grad_input`, `rmsnorm_grad_gamma`, `cross_entropy_logits_grad`, `embedding_scatter_accum`, matmul backward via existing `matmul` + `transpose` dtype variants).
   - Wire dtype tolerances into the autograd gradient-check tests.
   - Extend `AdamW` with `f32` master weights and the `cast_f32_to_*` fused kernel.
   - CLI: `llm train --dtype bf16` runs end-to-end on `tiny`/`small` configs.
   - Success: `tiny` trained for N steps in bf16 reaches loss within X of `f32` baseline (X documented in tasks).

3. **Checkpoint round-trip slice.**
   - Loader: read per-tensor dtype, optionally upcast with explicit flag.
   - Writer: persist the parameter's compute dtype (or its master dtype, configurable).
   - Inspect command: report dtype per tensor.
   - Success: `train → save → load → generate` round-trips in bf16 with no `f32` intermediate on disk.

**Rollback strategy:** Because the `f32` path is byte-identical, the rollback is "stop passing `--dtype` from the CLI and stop calling the new constructors with a non-`f32` dtype". The new kernels can stay registered (cheap pipeline objects) or be removed by reverting `registerCoreKernels` additions. Checkpoints written with explicit `dtype: 'f32'` remain readable by older versions because the field defaults to `'f32'` on read.

## Open Questions

1. **Do we want `cast` to be differentiable** (e.g. so a future user can train with `bf16` compute but recover `f32` gradients into the master weight via autograd, instead of via the `AdamW`-internal write-back)? Current decision: no, `AdamW` owns the master-weight round-trip and `cast` is forward-only. Revisit if a non-AdamW optimizer lands.

2. **Should the kernel ID suffix be `.bf16` / `.fp16` or fold dtype into a struct?** Current decision: string suffix because the kernel registry is `Map<string, GPUComputePipeline>` and we don't want to widen the registry's value type just for dtype tagging. Revisit if we ever add per-kernel metadata beyond dtype.

3. **Bias / norm parameter dtype.** PyTorch keeps `LayerNorm` weights in `f32` even under autocast. Current decision: every parameter is `bf16` by default; norms / biases get the master-weight treatment like everything else. We will measure on `small` and revisit if norm parameters specifically destabilize.

4. **Threshold for the "measurable win" success criterion in `proposal.md`.** Current target: ≥ 1.5× VRAM saving on the matmul activations of `small` (`d_model = 128`, `seqLen = 256`) or ≥ 1.2× tokens/sec in training, whichever is reproducible first. To be finalized in `tasks.md` once we measure on the reference adapter.

5. **`shader-f16` adoption among Dawn versions we pin.** We currently pin `webgpu` (Dawn v0.4.0 per `docs/project.md`). Confirm at task time that the feature is exposed; if not, the fp16 path uses `unpack2x16float` exclusively and we revisit the bf16 packing helpers only.
