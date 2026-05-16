# dtype-16bit

## ADDED Requirements

### Requirement: Tensor dtype enum is widened to include bf16 and fp16

The implementation SHALL widen the `DType` type exported by `@llm-transformer-diy/tensor` to `'f32' | 'bf16' | 'fp16' | 'u32'`. The default `Tensor.dtype` for every existing call site that does not explicitly pass a dtype SHALL remain `'f32'`. The implementation SHALL NOT introduce any new `Device` value (still `'gpu'` only) and SHALL NOT make `Tensor` mutable; `Tensor.dtype` SHALL remain a `readonly` field on the record.

#### Scenario: Default dtype is f32 for existing constructors

- **WHEN** a consumer calls `createZeros(rt, shape)` or `createFromArray(rt, data, shape)` without passing a dtype argument
- **THEN** the returned `Tensor` has `dtype === 'f32'`

#### Scenario: Constructors accept bf16 and fp16 as trailing dtype argument

- **WHEN** a consumer calls `createZeros(rt, shape, 'bf16')` or `createZeros(rt, shape, 'fp16')` on a runtime whose adapter supports the requested dtype
- **THEN** the returned `Tensor` has `dtype` equal to the requested dtype and `length === productOfShape(shape)`

#### Scenario: u32 indices tensors are unchanged

- **WHEN** a consumer constructs an indices tensor via `createIndices` or `makeIndicesTensor`
- **THEN** the returned `Tensor` has `dtype === 'u32'` and is byte-identical to the pre-change behavior

---

### Requirement: GpuContext probes shader-f16 and exposes a per-dtype capability check

The implementation SHALL extend `GpuBootstrapOptions` with optional `requiredFeatures` and `optionalFeatures` lists forwarded to `adapter.requestDevice`. By default, `'shader-f16'` SHALL be requested as an optional feature. The implementation SHALL record the granted features on `GpuContext` as `grantedFeatures: ReadonlySet<GPUFeatureName>` and SHALL expose a free function or method `supportsDtype(ctx, dtype)` that returns `true` for every dtype the runtime can honor. `bf16` SHALL always be reported as supported (the implementation emulates it via `u16`-packed storage and in-shader expansion). `fp16` SHALL be reported as supported when either `shader-f16` was granted or the implementation's `unpack2x16float`-based fallback is enabled. `f32` and `u32` SHALL always be reported as supported.

#### Scenario: bf16 is always supported

- **WHEN** `supportsDtype(ctx, 'bf16')` is queried on any successfully constructed `GpuContext`
- **THEN** the result is `true`

#### Scenario: fp16 is supported when shader-f16 is granted

- **WHEN** `GpuContext.create` is called on an adapter that grants `shader-f16` and `supportsDtype(ctx, 'fp16')` is queried
- **THEN** the result is `true` and `ctx.grantedFeatures.has('shader-f16')` is `true`

#### Scenario: Requested-but-unsupported required feature fails loudly

- **WHEN** `GpuContext.create` is called with `requiredFeatures: ['shader-f16']` on an adapter that does not grant `shader-f16`
- **THEN** the call rejects with an `Error` whose message identifies the missing feature

---

### Requirement: CPU bit-level pack and unpack helpers cover bf16 and fp16

The implementation SHALL expose CPU pack/unpack helper functions from `@llm-transformer-diy/gpu`: `packBf16FromFloat32`, `unpackBf16ToFloat32`, `packFp16FromFloat32`, and `unpackFp16ToFloat32`. Pack helpers SHALL accept a `Float32Array` input plus a target `Uint16Array` (or return a fresh one) of matching length. Bf16 pack SHALL truncate the low 16 mantissa bits using round-to-nearest-even semantics; fp16 pack SHALL implement IEEE-754 binary16 round-to-nearest-even including subnormals and `±Inf` / `NaN`. Unpack helpers SHALL produce a `Float32Array` whose elements equal the dtype-rounded value of every input element.

#### Scenario: bf16 pack-then-unpack round-trips representable values exactly

- **WHEN** an input `Float32Array` whose elements are all exactly representable in bf16 (e.g. clean integers `0`, `1`, `2`, `1024`, `-3`) is packed and then unpacked
- **THEN** the output `Float32Array` equals the input bit-for-bit

#### Scenario: bf16 pack rounds half-way mantissas to even

- **WHEN** a value whose dropped 17th mantissa bit is set with all lower bits zero is packed
- **THEN** the packed bf16 word equals the input rounded so the low retained mantissa bit is `0`

#### Scenario: fp16 pack saturates beyond binary16 dynamic range

- **WHEN** a positive value greater than the binary16 maximum (e.g. `1e30`) is passed to `packFp16FromFloat32`
- **THEN** the corresponding packed word equals `+Inf` in binary16 (`0x7C00`) and `unpackFp16ToFloat32` of that word equals `Infinity`

#### Scenario: fp16 pack preserves NaN

- **WHEN** a `NaN` value is passed to `packFp16FromFloat32`
- **THEN** `unpackFp16ToFloat32` of the resulting word returns `NaN`

---

### Requirement: 16-bit storage buffer helpers allocate and round-trip dtype-sized payloads

The implementation SHALL expose `createBf16StorageBuffer(device, elementCount)` and `createFp16StorageBuffer(device, elementCount)` returning a `GPUBuffer` of byte length `elementCount * 2` rounded up to WGSL's 4-byte alignment when required. The implementation SHALL expose paired `writeBf16Buffer` / `writeFp16Buffer` (accepting a `Uint16Array` payload produced by the CPU pack helpers) and `readBf16Buffer` / `readFp16Buffer` (returning a `Uint16Array`). `bytesPerElement(dtype)` SHALL return `4` for `f32` / `u32` and `2` for `bf16` / `fp16`.

#### Scenario: Bf16 buffer round-trip preserves the packed payload

- **WHEN** the caller allocates a bf16 buffer of length `n`, writes a packed `Uint16Array` payload, and reads it back without dispatching any compute
- **THEN** the readback `Uint16Array` equals the written payload element-for-element

#### Scenario: bytesPerElement matches the dtype

- **WHEN** `bytesPerElement('bf16')` and `bytesPerElement('fp16')` are evaluated
- **THEN** both equal `2`, while `bytesPerElement('f32')` and `bytesPerElement('u32')` both equal `4`

---

### Requirement: Core WGSL kernels register bf16 and fp16 variants under suffixed IDs

For each kernel currently registered under `core.<name>` (matmul, add, mul, scale, gelu, gelu_backward, softmax, softmax_backward, rmsnorm, rmsnorm_grad_input, rmsnorm_grad_gamma, embedding_gather, embedding_scatter_accum, cross_entropy_from_logits, cross_entropy_logits_grad, reduce_sum, reduce_max, transpose, broadcast_fill_scalar), the implementation SHALL register two additional pipelines under the stable identifiers `core.<name>.bf16` and `core.<name>.fp16`. The existing `core.<name>` identifier SHALL retain exact `f32` semantics. Each variant SHALL bind storage in the matching dtype (bf16 via `u16` packed in `array<u32>`, fp16 via `array<f16>` when `shader-f16` is granted or `array<u32>` with `unpack2x16float` otherwise). Reductions and matmul inner accumulators SHALL accumulate into `f32` regardless of the input dtype, narrowing only on store.

#### Scenario: Bf16 matmul agrees with the f32 reference within bf16 tolerance

- **WHEN** tests dispatch `core.matmul.bf16` on small integer-valued operands with WebGPU available
- **THEN** the readback, unpacked to `Float32Array`, equals the CPU `f32` reference rounded element-wise to bf16, within a documented bf16-specific floating-point tolerance

#### Scenario: Fp16 matmul agrees with the f32 reference within fp16 tolerance

- **WHEN** tests dispatch `core.matmul.fp16` on small operands with WebGPU available
- **THEN** the readback, unpacked to `Float32Array`, equals the CPU `f32` reference rounded element-wise to fp16, within a documented fp16-specific floating-point tolerance

#### Scenario: f32 kernels remain registered under the unsuffixed identifier

- **WHEN** `KernelRegistry.get('core.matmul')` is queried after `registerCoreKernels` returns
- **THEN** the pipeline is the same `f32` pipeline registered before this change and its numerical behavior is unchanged

#### Scenario: Bf16 reduction accumulates in f32

- **WHEN** `core.reduce_sum.bf16` is dispatched on `n` bf16 operands whose `f32`-domain sum is finite but where pairwise bf16 accumulation would overflow or drop bits
- **THEN** the readback (unpacked to `f32`) is closer to the `f32` reference sum than to the bf16-accumulated sum, within the documented bf16 tolerance

---

### Requirement: Tensor op dispatch routes by dtype and rejects mixed-dtype operands

The op functions exported from `@llm-transformer-diy/tensor` (`matmul`, `add`, `mul`, `scale`, `gelu`, `geluBackward`, `transpose`, `softmax`, `softmaxBackward`, `rmsnorm`, `rmsnormGradInput`, `rmsnormGradGamma`, `embedding`, `scatterEmbeddingAccum`, `crossEntropyRowsFromLogits`, `crossEntropyLogitsGrad`, `sumFlat`, `broadcastFillFromScalar`, `copyF32Elems`, `adamWStepInPlace`) SHALL dispatch to the kernel matching the dtype of their tensor operands. All tensor operands of a single op call SHALL have the same `dtype`, and the op SHALL throw an `Error` synchronously, before any kernel dispatch, when dtypes disagree. Output tensors SHALL be allocated at the shared operand dtype. The functions SHALL retain their existing signatures (with `scale` accepting a `number` scalar regardless of operand dtype).

#### Scenario: matmul on bf16 operands produces a bf16 output

- **WHEN** `matmul(rt, a, b)` is called with both operands at `dtype === 'bf16'`
- **THEN** the returned tensor has `dtype === 'bf16'` and the kernel dispatched is `core.matmul.bf16`

#### Scenario: Mixed-dtype add throws before dispatch

- **WHEN** `add(rt, a, b)` is called with `a.dtype === 'bf16'` and `b.dtype === 'f32'`
- **THEN** the call throws an `Error` synchronously and references both dtypes; no kernel is dispatched

#### Scenario: f32 default path remains byte-identical

- **WHEN** any op function is called with operands at `dtype === 'f32'`
- **THEN** the dispatched kernel ID equals the existing `core.<name>` (no `.bf16` / `.fp16` suffix) and the numerical output is identical to the pre-change behavior

---

### Requirement: Explicit cast op converts between dtypes without gradient

The implementation SHALL expose a tensor-layer free function `cast(rt, t, dtype)` that returns a freshly allocated `Tensor` containing the elementwise dtype conversion of `t`. The op SHALL dispatch a dedicated WGSL kernel registered under `core.cast.<from>.<to>` (with at least the pairs `f32→bf16`, `bf16→f32`, `f32→fp16`, `fp16→f32` registered). When called with `t.dtype === dtype`, `cast` SHALL return a freshly allocated independent copy at the same dtype. The autograd layer SHALL NOT register a backward Operation for `cast` in this scope.

#### Scenario: cast round-trips f32 through bf16 with documented rounding loss

- **WHEN** a consumer casts a `f32` tensor to `bf16` and back via `cast`
- **THEN** the readback equals the input element-wise rounded to bf16 (lower 16 mantissa bits zeroed, RNE) within bf16 tolerance

#### Scenario: cast to same dtype copies the buffer

- **WHEN** `cast(rt, t, t.dtype)` is called
- **THEN** the returned tensor's `buffer` is not reference-equal to `t.buffer`, has the same `shape` and `dtype`, and contains the same data

---

### Requirement: Autograd allocates grad tensors at the forward dtype

The `@llm-transformer-diy/autograd` layer SHALL allocate `Variable.grad` at the same `dtype` as the variable's forward `tensor`. Every `Operation.backward(rt, gradOutput)` implementation SHALL produce `Tensor | null` entries whose dtype matches the corresponding input variable's dtype. Mixed-dtype accumulation across the graph SHALL be prohibited at the same boundary as forward dispatch (Decision 9). Gradient-check tests SHALL parameterize over `f32`, `bf16`, and `fp16`, using dtype-specific tolerances (looser for `bf16` and `fp16` than for `f32`).

#### Scenario: backward dtype matches forward dtype

- **WHEN** an autograd op's forward output has `dtype === 'bf16'` and `loss.backward()` runs through it
- **THEN** every gradient tensor accumulated onto its inputs has `dtype === 'bf16'`

#### Scenario: gradient-check passes for bf16 within bf16 tolerance

- **WHEN** the gradient-check harness runs on a small composed graph parameterized at `bf16`
- **THEN** the analytical gradients agree with the central-finite-difference gradients (computed in `f32` for the reference) within the documented bf16 tolerance

---

### Requirement: Mixed-precision training keeps an f32 master tensor and f32 Adam moments

The `Parameter` record in `@llm-transformer-diy/autograd` SHALL optionally carry a `masterTensor: Tensor | null` field. When present, `masterTensor.dtype` SHALL equal `'f32'`. `AdamW` SHALL operate on `masterTensor` (or `tensor` when no master is set), SHALL allocate its first and second moment tensors at `dtype === 'f32'` regardless of the compute parameter dtype, and SHALL after each step cast the updated `f32` master values back into the compute parameter `tensor` via the `core.cast.f32.<bf16|fp16>` kernel. `ParameterRegistry.zeroGrad`, `clipGradNorm`, and `totalParams` SHALL continue to operate as one parameter per registered entry regardless of whether a master tensor is present.

#### Scenario: Bf16 parameter with master keeps f32 moments

- **WHEN** an `AdamW` instance is constructed over a `ParameterRegistry` whose parameters have `tensor.dtype === 'bf16'` and `masterTensor.dtype === 'f32'`
- **THEN** the moment tensors allocated by the optimizer for those parameters have `dtype === 'f32'`

#### Scenario: AdamW step writes back through the cast kernel

- **WHEN** `AdamW.step()` is called on a parameter with a `bf16` compute tensor and an `f32` master tensor
- **THEN** the moment update happens in `f32`, the master tensor's `f32` buffer is updated in place, and the bf16 compute tensor receives a fresh cast of the updated master values

#### Scenario: Pure f32 parameters are unaffected

- **WHEN** an `AdamW` instance steps over parameters with no `masterTensor` and `tensor.dtype === 'f32'`
- **THEN** the moment dtypes, update kernel, and numerical results are identical to the pre-change behavior

---

### Requirement: Checkpoint sidecar carries per-tensor dtype and is backward compatible

The `@llm-transformer-diy/io` checkpoint sidecar's `tensors[name]` entries SHALL accept and persist a `dtype` field with values restricted to `'f32' | 'bf16' | 'fp16'`. The binary payload offset and stride for each tensor SHALL be computed as `productOfShape(shape) * bytesPerElement(dtype)`. When reading a sidecar that omits `dtype` for a tensor, the loader SHALL interpret that tensor as `'f32'`. When the runtime cannot satisfy the on-disk dtype (per `supportsDtype`), the loader SHALL throw an `Error` whose message names the tensor and the offending dtype, unless the caller passes an explicit "allow upcast to f32" option.

#### Scenario: Existing checkpoints (no dtype field) load as f32

- **WHEN** a checkpoint sidecar produced before this change (no `dtype` field on its tensor entries) is loaded
- **THEN** every tensor is reconstructed as `Float32Array` of length `productOfShape(shape)` and the loaded model is numerically identical to the pre-change behavior

#### Scenario: Bf16 round-trip preserves the packed payload

- **WHEN** a model whose parameters are at `dtype === 'bf16'` is saved and then loaded into a fresh model
- **THEN** the loaded compute tensor's GPU buffer, read back and unpacked, equals the pre-save tensor's read-back-and-unpacked values bit-for-bit

#### Scenario: Loader refuses unsupported dtype without explicit upcast

- **WHEN** a checkpoint declares `dtype === 'fp16'` for some tensor and the runtime's `supportsDtype('fp16')` is `false`, and the caller passes no upcast option
- **THEN** the loader throws an `Error` naming the tensor and the unsupported dtype, and does not allocate a partial model

---

### Requirement: CLI exposes an opt-in --dtype flag and fails loudly when unsupported

The `llm` CLI's `train` and `generate` subcommands SHALL accept an optional `--dtype <f32|bf16|fp16>` flag, defaulting to `'f32'`. When the flag is omitted, the CLI SHALL behave byte-identically to its pre-change behavior. When the flag is set to a non-`f32` value, the CLI SHALL probe the adapter via `supportsDtype` immediately after `GpuContext.create` and SHALL exit with a non-zero status and a clear human-readable error if the requested dtype is unsupported. When `train --dtype fp16` is requested, the CLI SHALL emit a single-line warning that fp16 training is unstable without loss scaling and recommend `bf16`.

#### Scenario: Default invocation uses f32

- **WHEN** the user runs `llm generate --model ./model/` (or `llm train --data … --config tiny`) with no `--dtype` flag
- **THEN** every allocation, kernel dispatch, and checkpoint read goes through the `f32` path and the command behaves identically to the pre-change behavior

#### Scenario: Bf16 training is accepted when supported

- **WHEN** the user runs `llm train --data … --config tiny --dtype bf16` on an adapter where `supportsDtype('bf16')` is `true`
- **THEN** the model parameters are created at `bf16` with `f32` master weights, `AdamW` is constructed with `f32` moments, and the training loop completes a step without error

#### Scenario: Unsupported dtype exits non-zero with a clear error

- **WHEN** the user runs `llm generate --model ./model/ --dtype fp16` on an adapter where `supportsDtype('fp16')` is `false`
- **THEN** the process exits with a non-zero status code, prints a message naming the adapter and the unsupported dtype, and does not load model weights

#### Scenario: Train --dtype fp16 emits a stability warning

- **WHEN** the user runs `llm train --data … --dtype fp16` on a supported adapter
- **THEN** the CLI prints a single warning line referencing fp16 training instability and recommending bf16, then continues

---

### Requirement: WebGPU-gated tests for 16-bit dtypes remain non-fatal without an adapter

Automated tests that exercise the bf16 / fp16 paths on the GPU SHALL follow the same policy as `gpu-bootstrap`, `core-wgsl-kernels`, `reduction-norm-kernels`, and `tensor-bootstrap`: if no usable WebGPU adapter is available, GPU-specific dtype tests SHALL be skipped or otherwise gated so the package test command still exits successfully. CPU-side pack/unpack helper tests SHALL run regardless of WebGPU availability.

#### Scenario: Suite passes without WebGPU

- **WHEN** the package test suite runs in an environment without a usable WebGPU adapter
- **THEN** GPU-dependent bf16 / fp16 tests do not fail the suite solely due to missing GPU hardware or stack, while the pure-CPU pack/unpack and bytesPerElement tests still run and pass

#### Scenario: Suite skips fp16 GPU tests on adapters without fp16 support

- **WHEN** the package test suite runs against an adapter where `supportsDtype('fp16')` is `false`
- **THEN** the fp16 GPU tests are skipped (not failed) and the bf16 tests still run and pass
