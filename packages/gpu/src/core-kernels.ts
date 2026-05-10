import type { GpuContext } from './device';
import type { KernelRegistry } from './kernel-registry';
import {
  loadAdamWStepWgsl,
  loadBroadcastFillScalarWgsl,
  loadCrossEntropyFromLogitsWgsl,
  loadCrossEntropyLogitsGradWgsl,
  loadElementwiseAddWgsl,
  loadElementwiseGeluBackwardWgsl,
  loadElementwiseGeluWgsl,
  loadElementwiseMulWgsl,
  loadElementwiseScaleWgsl,
  loadEmbeddingGatherWgsl,
  loadEmbeddingScatterAccumWgsl,
  loadMatMulWgsl,
  loadReduceMaxWgsl,
  loadReduceSumWgsl,
  loadRmsNormGradGammaWgsl,
  loadRmsNormGradInputWgsl,
  loadRmsNormWgsl,
  loadSoftmaxBackwardWgsl,
  loadSoftmaxWgsl,
  loadTransposeWgsl,
} from './kernels/load-core-wgsl';
import { type ComputeBindingDesc, createSimpleComputePipeline } from './pipeline';

/** Naive matmul: row-major A (M×K), B (K×N), output C (M×N). */
export const CORE_KERNEL_MATMUL = 'core.matmul';

/** Row-major transpose: src rows×cols → dst cols×rows. */
export const CORE_KERNEL_TRANSPOSE = 'core.transpose';

/** Elementwise add on equal-length 1D f32 buffers. */
export const CORE_KERNEL_ADD = 'core.add';

/** Elementwise multiply on equal-length 1D f32 buffers. */
export const CORE_KERNEL_MUL = 'core.mul';

/** out[i] = a[i] * scale (uniform scalar). */
export const CORE_KERNEL_SCALE = 'core.scale';

/** GELU activation per element (tanh formulation; see cpu-reference.ts). */
export const CORE_KERNEL_GELU = 'core.gelu';

/** Upstream gradient times GELU derivative w.r.t. forward input (elementwise). */
export const CORE_KERNEL_GELU_BACKWARD = 'core.gelu_backward';

/** Row-wise sum of row-major [rows, cols] f32 matrix → [rows] f32 vector. */
export const CORE_KERNEL_REDUCE_SUM = 'core.reduce_sum';

/** Row-wise max of row-major [rows, cols] f32 matrix → [rows] f32 vector. */
export const CORE_KERNEL_REDUCE_MAX = 'core.reduce_max';

/** Numerically-stable row-wise softmax of [rows, cols] f32 matrix (two-pass per row). */
export const CORE_KERNEL_SOFTMAX = 'core.softmax';

/** Jacobian–vector product for row-wise softmax (grad w.r.t. logits). */
export const CORE_KERNEL_SOFTMAX_BACKWARD = 'core.softmax_backward';

/** Row-wise RMSNorm of [rows, cols] f32 matrix with learned `gamma: [cols]`. */
export const CORE_KERNEL_RMSNORM = 'core.rmsnorm';

/** ∂L/∂x for RMSNorm (reads γ, grad_out; writes grad_input). */
export const CORE_KERNEL_RMSNORM_GRAD_INPUT = 'core.rmsnorm_grad_input';

/** ∂L/∂γ for RMSNorm (column reductions). */
export const CORE_KERNEL_RMSNORM_GRAD_GAMMA = 'core.rmsnorm_grad_gamma';

/** Row-major gather of `[vocab, embed]` by u32 indices → `[tokens, embed]`. */
export const CORE_KERNEL_EMBEDDING_GATHER = 'core.embedding_gather';

/**
 * Atomic accumulation of `[tokens, embed]` token gradients into `[vocab, embed]` dense
 * grad backing (must start zero-filled).
 */
export const CORE_KERNEL_EMBEDDING_SCATTER_ACCUM = 'core.embedding_scatter_accum';

/** Stable per-example cross-entropy from raw logits `[rows, classes]`. */
export const CORE_KERNEL_CROSS_ENTROPY_FROM_LOGITS = 'core.cross_entropy_from_logits';

/** ∂L/∂logits given softmax probs and upstream per-row grads. */
export const CORE_KERNEL_CROSS_ENTROPY_LOGITS_GRAD = 'core.cross_entropy_logits_grad';

/** `out[i] = scalar[0]` for length `n` (broadcast from a `[1]` buffer). */
export const CORE_KERNEL_BROADCAST_FILL_SCALAR = 'core.broadcast_fill_scalar';

/** Fused AdamW: in-place θ, m, v; read-only gradient buffer. */
export const CORE_KERNEL_ADAMW_STEP = 'core.adamw_step';

/** All core numeric kernels registered by {@link registerCoreKernels}. */
export const CORE_KERNEL_IDS = [
  CORE_KERNEL_MATMUL,
  CORE_KERNEL_TRANSPOSE,
  CORE_KERNEL_ADD,
  CORE_KERNEL_MUL,
  CORE_KERNEL_SCALE,
  CORE_KERNEL_GELU,
  CORE_KERNEL_GELU_BACKWARD,
  CORE_KERNEL_REDUCE_SUM,
  CORE_KERNEL_REDUCE_MAX,
  CORE_KERNEL_SOFTMAX,
  CORE_KERNEL_SOFTMAX_BACKWARD,
  CORE_KERNEL_RMSNORM,
  CORE_KERNEL_RMSNORM_GRAD_INPUT,
  CORE_KERNEL_RMSNORM_GRAD_GAMMA,
  CORE_KERNEL_EMBEDDING_GATHER,
  CORE_KERNEL_EMBEDDING_SCATTER_ACCUM,
  CORE_KERNEL_CROSS_ENTROPY_FROM_LOGITS,
  CORE_KERNEL_CROSS_ENTROPY_LOGITS_GRAD,
  CORE_KERNEL_BROADCAST_FILL_SCALAR,
  CORE_KERNEL_ADAMW_STEP,
] as const;

const WGSL_MAIN = 'main';
const WORKGROUP_SIZE = 256;
/** All core kernel uniforms fit in a 16-byte block (vec4-sized). */
const UNIFORM_BYTES = 16;

/** `adamw-step.wgsl` — two vec4<u32> + two vec4<f32> blocks. */
const ADAM_UNIFORM_BYTES = 48;

type Kind = 'uniform' | 'read-only-storage' | 'storage';

function bindings(...kinds: Kind[]): ComputeBindingDesc[] {
  return kinds.map((type, binding) => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type },
  }));
}

function workgroupCount(elementCount: number): number {
  return Math.ceil(elementCount / WORKGROUP_SIZE);
}

/**
 * Compiles all core WGSL kernels and registers them on `registry` under their
 * `CORE_KERNEL_*` IDs. Pipelines reuse `createSimpleComputePipeline`; binding
 * layout per kernel is documented in the `bindings(...)` calls below.
 */
export function registerCoreKernels(device: GPUDevice, registry: KernelRegistry): void {
  const make = (code: string, kinds: Kind[]): GPUComputePipeline =>
    createSimpleComputePipeline(device, {
      code,
      entryPoint: WGSL_MAIN,
      bindings: bindings(...kinds),
    }).pipeline;

  // matmul: uniform(dims), readonly A, readonly B, storage C
  registry.register(
    CORE_KERNEL_MATMUL,
    make(loadMatMulWgsl(), ['uniform', 'read-only-storage', 'read-only-storage', 'storage']),
  );

  // transpose: uniform(rows,cols), readonly src, storage dst
  registry.register(
    CORE_KERNEL_TRANSPOSE,
    make(loadTransposeWgsl(), ['uniform', 'read-only-storage', 'storage']),
  );

  // elementwise binary (add, mul): uniform(n), readonly a, readonly b, storage out
  const ewBinaryKinds: Kind[] = ['uniform', 'read-only-storage', 'read-only-storage', 'storage'];
  registry.register(CORE_KERNEL_ADD, make(loadElementwiseAddWgsl(), ewBinaryKinds));
  registry.register(CORE_KERNEL_MUL, make(loadElementwiseMulWgsl(), ewBinaryKinds));

  // elementwise unary (scale, gelu): uniform(n[, scale]), readonly in, storage out
  const ewUnaryKinds: Kind[] = ['uniform', 'read-only-storage', 'storage'];
  registry.register(CORE_KERNEL_SCALE, make(loadElementwiseScaleWgsl(), ewUnaryKinds));
  registry.register(CORE_KERNEL_GELU, make(loadElementwiseGeluWgsl(), ewUnaryKinds));

  // elementwise gelu backward: uniform(n), readonly grad_out, readonly inp, storage grad_in
  const ewGeluBackwardKinds: Kind[] = [
    'uniform',
    'read-only-storage',
    'read-only-storage',
    'storage',
  ];
  registry.register(
    CORE_KERNEL_GELU_BACKWARD,
    make(loadElementwiseGeluBackwardWgsl(), ewGeluBackwardKinds),
  );

  // row-wise reductions / softmax: uniform(rows,cols), readonly input, storage out
  const reduceKinds: Kind[] = ['uniform', 'read-only-storage', 'storage'];
  registry.register(CORE_KERNEL_REDUCE_SUM, make(loadReduceSumWgsl(), reduceKinds));
  registry.register(CORE_KERNEL_REDUCE_MAX, make(loadReduceMaxWgsl(), reduceKinds));
  registry.register(CORE_KERNEL_SOFTMAX, make(loadSoftmaxWgsl(), reduceKinds));

  // softmax backward: uniform(rows,cols), readonly grad_out, readonly y, storage grad_in
  const softmaxBackwardKinds: Kind[] = [
    'uniform',
    'read-only-storage',
    'read-only-storage',
    'storage',
  ];
  registry.register(
    CORE_KERNEL_SOFTMAX_BACKWARD,
    make(loadSoftmaxBackwardWgsl(), softmaxBackwardKinds),
  );

  // rmsnorm: uniform(rows,cols,epsilon), readonly input, readonly gamma, storage out
  registry.register(
    CORE_KERNEL_RMSNORM,
    make(loadRmsNormWgsl(), ['uniform', 'read-only-storage', 'read-only-storage', 'storage']),
  );

  const rmsNormBackwardKinds: Kind[] = [
    'uniform',
    'read-only-storage',
    'read-only-storage',
    'read-only-storage',
    'storage',
  ];
  registry.register(
    CORE_KERNEL_RMSNORM_GRAD_INPUT,
    make(loadRmsNormGradInputWgsl(), rmsNormBackwardKinds),
  );

  registry.register(
    CORE_KERNEL_RMSNORM_GRAD_GAMMA,
    make(loadRmsNormGradGammaWgsl(), ['uniform', 'read-only-storage', 'read-only-storage', 'storage']),
  );

  // embedding_gather: uniform(num_tokens, embed_dim,...), weights, indices u32, out
  registry.register(
    CORE_KERNEL_EMBEDDING_GATHER,
    make(loadEmbeddingGatherWgsl(), [
      'uniform',
      'read-only-storage',
      'read-only-storage',
      'storage',
    ]),
  );

  // embedding_scatter_accum: grad_tokens + indices readonly; grad_weights_atomic read-write
  registry.register(
    CORE_KERNEL_EMBEDDING_SCATTER_ACCUM,
    make(loadEmbeddingScatterAccumWgsl(), [
      'uniform',
      'read-only-storage',
      'read-only-storage',
      'storage',
    ]),
  );

  // cross_entropy_forward: logits, targets(u32), per-row losses
  registry.register(
    CORE_KERNEL_CROSS_ENTROPY_FROM_LOGITS,
    make(loadCrossEntropyFromLogitsWgsl(), [
      'uniform',
      'read-only-storage',
      'read-only-storage',
      'storage',
    ]),
  );

  // cross_entropy logits grad: probs, targets, upstream_per_row, grad_logits
  registry.register(
    CORE_KERNEL_CROSS_ENTROPY_LOGITS_GRAD,
    make(loadCrossEntropyLogitsGradWgsl(), [
      'uniform',
      'read-only-storage',
      'read-only-storage',
      'read-only-storage',
      'storage',
    ]),
  );

  registry.register(
    CORE_KERNEL_BROADCAST_FILL_SCALAR,
    make(loadBroadcastFillScalarWgsl(), ['uniform', 'read-only-storage', 'storage']),
  );

  registry.register(
    CORE_KERNEL_ADAMW_STEP,
    make(loadAdamWStepWgsl(), [
      'uniform',
      'storage',
      'read-only-storage',
      'storage',
      'storage',
    ]),
  );
}

/**
 * Allocates a transient 16-byte uniform buffer, fills it, and uploads to the GPU.
 * The caller is responsible for `destroy()` after submission. Per WebGPU spec,
 * `destroy()` may be called while submissions are still in flight — the GPU keeps
 * the underlying allocation alive until the queue drains.
 */
function makeUniform(ctx: GpuContext, fill: (view: DataView) => void): GPUBuffer {
  const ab = new ArrayBuffer(UNIFORM_BYTES);
  fill(new DataView(ab));
  const buf = ctx.device.createBuffer({
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  ctx.queue.writeBuffer(buf, 0, ab);
  return buf;
}

function makeAdamUniform(
  ctx: GpuContext,
  elementCount: number,
  hyper: {
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
    biasCorrection1: number;
    biasCorrection2: number;
  },
): GPUBuffer {
  if (!Number.isInteger(elementCount) || elementCount <= 0) {
    throw new Error(`dispatchAdamWStep: elementCount must be positive (got ${String(elementCount)})`);
  }
  const ab = new ArrayBuffer(ADAM_UNIFORM_BYTES);
  const v = new DataView(ab);
  v.setUint32(0, elementCount, true);
  v.setUint32(4, 0, true);
  v.setUint32(8, 0, true);
  v.setUint32(12, 0, true);
  v.setFloat32(16, hyper.lr, true);
  v.setFloat32(20, hyper.beta1, true);
  v.setFloat32(24, hyper.beta2, true);
  v.setFloat32(28, hyper.eps, true);
  v.setFloat32(32, hyper.weightDecay, true);
  v.setFloat32(36, hyper.biasCorrection1, true);
  v.setFloat32(40, hyper.biasCorrection2, true);
  v.setFloat32(44, 0, true);
  const buf = ctx.device.createBuffer({
    size: ADAM_UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  ctx.queue.writeBuffer(buf, 0, ab);
  return buf;
}

function bg(binding: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding, resource: { buffer } };
}

/**
 * Builds a bind group on `pipeline.getBindGroupLayout(0)`, encodes one compute
 * pass with `entries`, and submits. Synchronous — caller awaits a readback for
 * downstream sync, or chains more dispatches before any host-side wait.
 */
function dispatchKernel(
  ctx: GpuContext,
  pipeline: GPUComputePipeline,
  entries: GPUBindGroupEntry[],
  totalThreads: number,
): void {
  const bindGroup = ctx.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const encoder = ctx.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupCount(totalThreads));
  pass.end();
  ctx.queue.submit([encoder.finish()]);
}

/**
 * Dispatches matmul: **row-major** A (M×K), B (K×N), writes **row-major** C (M×N).
 * Uniform packs `M, N, K` as the first three u32 of a vec4 (fourth slot padding).
 */
export function dispatchMatMul(
  ctx: GpuContext,
  registry: KernelRegistry,
  bufferA: GPUBuffer,
  bufferB: GPUBuffer,
  bufferC: GPUBuffer,
  M: number,
  N: number,
  K: number,
): void {
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, M, true);
    v.setUint32(4, N, true);
    v.setUint32(8, K, true);
    v.setUint32(12, 0, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_MATMUL),
    [bg(0, u), bg(1, bufferA), bg(2, bufferB), bg(3, bufferC)],
    M * N,
  );
  u.destroy();
}

/**
 * Transposes row-major `src` of shape rows×cols into row-major `dst` of shape
 * cols×rows. `src` and `dst` are both rows*cols floats; uniform packs `rows, cols`
 * as the first two u32 of a vec4 (rest padding).
 */
export function dispatchTranspose(
  ctx: GpuContext,
  registry: KernelRegistry,
  src: GPUBuffer,
  dst: GPUBuffer,
  rows: number,
  cols: number,
): void {
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, rows, true);
    v.setUint32(4, cols, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_TRANSPOSE),
    [bg(0, u), bg(1, src), bg(2, dst)],
    rows * cols,
  );
  u.destroy();
}

function dispatchElementwiseBinary(
  ctx: GpuContext,
  registry: KernelRegistry,
  kernelId: typeof CORE_KERNEL_ADD | typeof CORE_KERNEL_MUL,
  a: GPUBuffer,
  b: GPUBuffer,
  out: GPUBuffer,
  elementCount: number,
): void {
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, elementCount, true);
  });
  dispatchKernel(
    ctx,
    registry.get(kernelId),
    [bg(0, u), bg(1, a), bg(2, b), bg(3, out)],
    elementCount,
  );
  u.destroy();
}

/** Elementwise add: `out[i] = a[i] + b[i]` for `elementCount` floats. */
export function dispatchAdd(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: GPUBuffer,
  b: GPUBuffer,
  out: GPUBuffer,
  elementCount: number,
): void {
  dispatchElementwiseBinary(ctx, registry, CORE_KERNEL_ADD, a, b, out, elementCount);
}

/** Elementwise multiply: `out[i] = a[i] * b[i]`. */
export function dispatchMul(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: GPUBuffer,
  b: GPUBuffer,
  out: GPUBuffer,
  elementCount: number,
): void {
  dispatchElementwiseBinary(ctx, registry, CORE_KERNEL_MUL, a, b, out, elementCount);
}

/** `out[i] = a[i] * scale`. Uniform packs `element_count: u32, scale: f32` + padding. */
export function dispatchScale(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: GPUBuffer,
  out: GPUBuffer,
  elementCount: number,
  scale: number,
): void {
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, elementCount, true);
    v.setFloat32(4, scale, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_SCALE),
    [bg(0, u), bg(1, a), bg(2, out)],
    elementCount,
  );
  u.destroy();
}

/** Applies GELU to each element: `out[i] = gelu(inp[i])`. */
export function dispatchGelu(
  ctx: GpuContext,
  registry: KernelRegistry,
  inp: GPUBuffer,
  out: GPUBuffer,
  elementCount: number,
): void {
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, elementCount, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_GELU),
    [bg(0, u), bg(1, inp), bg(2, out)],
    elementCount,
  );
  u.destroy();
}

/**
 * GELU backward: `grad_in[i] = grad_out[i] * gelu′(inp[i])` (same tanh
 * formulation as {@link dispatchGelu}).
 */
export function dispatchGeluBackward(
  ctx: GpuContext,
  registry: KernelRegistry,
  gradOut: GPUBuffer,
  inp: GPUBuffer,
  gradIn: GPUBuffer,
  elementCount: number,
): void {
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, elementCount, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_GELU_BACKWARD),
    [bg(0, u), bg(1, gradOut), bg(2, inp), bg(3, gradIn)],
    elementCount,
  );
  u.destroy();
}

function assertPositiveDims(label: string, rows: number, cols: number): void {
  if (!Number.isInteger(rows) || rows <= 0) {
    throw new Error(`${label}: rows must be a positive integer (got ${String(rows)})`);
  }
  if (!Number.isInteger(cols) || cols <= 0) {
    throw new Error(`${label}: cols must be a positive integer (got ${String(cols)})`);
  }
}

function dispatchRowwiseReduce(
  ctx: GpuContext,
  registry: KernelRegistry,
  kernelId: typeof CORE_KERNEL_REDUCE_SUM | typeof CORE_KERNEL_REDUCE_MAX,
  input: GPUBuffer,
  out: GPUBuffer,
  rows: number,
  cols: number,
): void {
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, rows, true);
    v.setUint32(4, cols, true);
  });
  dispatchKernel(ctx, registry.get(kernelId), [bg(0, u), bg(1, input), bg(2, out)], rows);
  u.destroy();
}

/**
 * Row-wise sum: `out[i] = Σ_j input[i*cols + j]`. `input` is row-major
 * `[rows, cols]` (length `rows*cols`); `out` has length `rows`. One thread per
 * row; throws synchronously if either dimension is non-positive before any GPU
 * work is queued.
 */
export function dispatchReduceSum(
  ctx: GpuContext,
  registry: KernelRegistry,
  input: GPUBuffer,
  out: GPUBuffer,
  rows: number,
  cols: number,
): void {
  assertPositiveDims('dispatchReduceSum', rows, cols);
  dispatchRowwiseReduce(ctx, registry, CORE_KERNEL_REDUCE_SUM, input, out, rows, cols);
}

/**
 * Row-wise max: `out[i] = max_j input[i*cols + j]`. Same shape contract as
 * {@link dispatchReduceSum}. The kernel initializes its per-row accumulator to
 * `-Infinity` so that an empty-cols row would yield `-Infinity` rather than a
 * stale value (the dispatch helper rejects `cols <= 0` regardless).
 */
export function dispatchReduceMax(
  ctx: GpuContext,
  registry: KernelRegistry,
  input: GPUBuffer,
  out: GPUBuffer,
  rows: number,
  cols: number,
): void {
  assertPositiveDims('dispatchReduceMax', rows, cols);
  dispatchRowwiseReduce(ctx, registry, CORE_KERNEL_REDUCE_MAX, input, out, rows, cols);
}

/**
 * Numerically stable row-wise softmax of `[rows, cols]`. The kernel runs the
 * textbook two-pass per row (subtract row max, exp, normalize) so rows with
 * large-magnitude values stay finite. `out` has the same shape as `input`.
 */
export function dispatchSoftmax(
  ctx: GpuContext,
  registry: KernelRegistry,
  input: GPUBuffer,
  out: GPUBuffer,
  rows: number,
  cols: number,
): void {
  assertPositiveDims('dispatchSoftmax', rows, cols);
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, rows, true);
    v.setUint32(4, cols, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_SOFTMAX),
    [bg(0, u), bg(1, input), bg(2, out)],
    rows,
  );
  u.destroy();
}

/**
 * Softmax backward: given upstream gradient w.r.t. softmax probabilities
 * `grad_out` and forward output `y` (same shape `[rows, cols]`), writes the
 * gradient w.r.t. logits into `grad_in`.
 */
export function dispatchSoftmaxBackward(
  ctx: GpuContext,
  registry: KernelRegistry,
  gradOut: GPUBuffer,
  y: GPUBuffer,
  gradIn: GPUBuffer,
  rows: number,
  cols: number,
): void {
  assertPositiveDims('dispatchSoftmaxBackward', rows, cols);
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, rows, true);
    v.setUint32(4, cols, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_SOFTMAX_BACKWARD),
    [bg(0, u), bg(1, gradOut), bg(2, y), bg(3, gradIn)],
    rows,
  );
  u.destroy();
}

/**
 * Row-wise RMSNorm of `[rows, cols]` with learned scale `gamma` of length
 * `cols`. Computes `out[i,j] = input[i,j] * gamma[j] / sqrt(mean(input[i,*]²) + epsilon)`.
 * Default `epsilon` matches `docs/project.md` (`1e-6`). Passing `epsilon === 0`
 * will divide by zero on a zero-norm row and is the caller's responsibility.
 *
 * `gammaLength` is the declared length of the `gamma` buffer in `f32`
 * elements; it is required so this helper can reject `gamma.length !== cols`
 * synchronously, before any GPU dispatch (the raw `GPUBuffer` does not carry
 * an element count).
 */
export function dispatchRmsNorm(
  ctx: GpuContext,
  registry: KernelRegistry,
  input: GPUBuffer,
  gamma: GPUBuffer,
  gammaLength: number,
  out: GPUBuffer,
  rows: number,
  cols: number,
  epsilon = 1e-6,
): void {
  assertPositiveDims('dispatchRmsNorm', rows, cols);
  if (gammaLength !== cols) {
    throw new Error(
      `dispatchRmsNorm: gamma length (${String(gammaLength)}) must equal cols (${String(cols)})`,
    );
  }
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, rows, true);
    v.setUint32(4, cols, true);
    v.setFloat32(8, epsilon, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_RMSNORM),
    [bg(0, u), bg(1, input), bg(2, gamma), bg(3, out)],
    rows,
  );
  u.destroy();
}

/** ∂L/∂input for RMSNorm; writes **`gradInput`** (same shape as **`input`**). */
export function dispatchRmsNormGradInput(
  ctx: GpuContext,
  registry: KernelRegistry,
  input: GPUBuffer,
  gamma: GPUBuffer,
  gammaLength: number,
  gradOut: GPUBuffer,
  gradInput: GPUBuffer,
  rows: number,
  cols: number,
  epsilon = 1e-6,
): void {
  assertPositiveDims('dispatchRmsNormGradInput', rows, cols);
  if (gammaLength !== cols) {
    throw new Error(
      `dispatchRmsNormGradInput: gamma length (${String(gammaLength)}) must equal cols (${String(cols)})`,
    );
  }
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, rows, true);
    v.setUint32(4, cols, true);
    v.setFloat32(8, epsilon, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_RMSNORM_GRAD_INPUT),
    [bg(0, u), bg(1, input), bg(2, gamma), bg(3, gradOut), bg(4, gradInput)],
    rows,
  );
  u.destroy();
}

/** ∂L/∂γ (`[cols]`); **`gradGamma`** must be zero-initialized or overwritten entirely. */
export function dispatchRmsNormGradGamma(
  ctx: GpuContext,
  registry: KernelRegistry,
  input: GPUBuffer,
  gradOut: GPUBuffer,
  gradGamma: GPUBuffer,
  rows: number,
  cols: number,
  epsilon = 1e-6,
): void {
  assertPositiveDims('dispatchRmsNormGradGamma', rows, cols);
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, rows, true);
    v.setUint32(4, cols, true);
    v.setFloat32(8, epsilon, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_RMSNORM_GRAD_GAMMA),
    [bg(0, u), bg(1, input), bg(2, gradOut), bg(3, gradGamma)],
    cols,
  );
  u.destroy();
}

/**
 * Embedding gather: reads weight table **`[vocab, embed]`** row-major (`weights`),
 * gathers rows indicated by **`indices`** (length `numTokens`). Writes **`out`** of
 * length `numTokens * embedDim` (**`[numTokens, embed]`** contiguous).
 */
export function dispatchEmbeddingGather(
  ctx: GpuContext,
  registry: KernelRegistry,
  weights: GPUBuffer,
  indices: GPUBuffer,
  out: GPUBuffer,
  numTokens: number,
  embedDim: number,
): void {
  if (!Number.isInteger(numTokens) || numTokens <= 0) {
    throw new Error(
      `dispatchEmbeddingGather: numTokens must be positive (got ${String(numTokens)})`,
    );
  }
  if (!Number.isInteger(embedDim) || embedDim <= 0) {
    throw new Error(`dispatchEmbeddingGather: embedDim must be positive (got ${String(embedDim)})`);
  }
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, numTokens, true);
    v.setUint32(4, embedDim, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_EMBEDDING_GATHER),
    [bg(0, u), bg(1, weights), bg(2, indices), bg(3, out)],
    numTokens * embedDim,
  );
  u.destroy();
}

/**
 * Scatter **`[tokens, embed]`** token gradients into **`[vocab, embed]`**. The **`gradWeights`**
 * buffer MUST start zero‑filled (CAS float atomics).
 */
export function dispatchEmbeddingScatterAccum(
  ctx: GpuContext,
  registry: KernelRegistry,
  gradTokens: GPUBuffer,
  indices: GPUBuffer,
  gradWeights: GPUBuffer,
  numTokens: number,
  embedDim: number,
): void {
  if (!Number.isInteger(numTokens) || numTokens <= 0) {
    throw new Error(
      `dispatchEmbeddingScatterAccum: numTokens must be positive (got ${String(numTokens)})`,
    );
  }
  if (!Number.isInteger(embedDim) || embedDim <= 0) {
    throw new Error(
      `dispatchEmbeddingScatterAccum: embedDim must be positive (got ${String(embedDim)})`,
    );
  }
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, numTokens, true);
    v.setUint32(4, embedDim, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_EMBEDDING_SCATTER_ACCUM),
    [bg(0, u), bg(1, gradTokens), bg(2, indices), bg(3, gradWeights)],
    numTokens * embedDim,
  );
  u.destroy();
}

/** Per‑row softmax cross‑entropy losses from logits and u32 targets. */
export function dispatchCrossEntropyFromLogits(
  ctx: GpuContext,
  registry: KernelRegistry,
  logits: GPUBuffer,
  targets: GPUBuffer,
  lossesOut: GPUBuffer,
  rows: number,
  cols: number,
): void {
  assertPositiveDims('dispatchCrossEntropyFromLogits', rows, cols);
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, rows, true);
    v.setUint32(4, cols, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_CROSS_ENTROPY_FROM_LOGITS),
    [bg(0, u), bg(1, logits), bg(2, targets), bg(3, lossesOut)],
    rows,
  );
  u.destroy();
}

/**
 * ∂Σ_k upstream_row[k] · CE(logits_row_k, target_k) / ∂logits given row‑wise **`probs`**
 * (**softmax logits**) per row.
 */
export function dispatchCrossEntropyLogitsGrad(
  ctx: GpuContext,
  registry: KernelRegistry,
  probs: GPUBuffer,
  targets: GPUBuffer,
  upstreamPerRow: GPUBuffer,
  gradLogitsOut: GPUBuffer,
  rows: number,
  cols: number,
): void {
  assertPositiveDims('dispatchCrossEntropyLogitsGrad', rows, cols);
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, rows, true);
    v.setUint32(4, cols, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_CROSS_ENTROPY_LOGITS_GRAD),
    [bg(0, u), bg(1, probs), bg(2, targets), bg(3, upstreamPerRow), bg(4, gradLogitsOut)],
    rows * cols,
  );
  u.destroy();
}

/** Writes **`out[i] = scalar[0]`** for `i ∈ [0, n)`. */
export function dispatchBroadcastFillScalar(
  ctx: GpuContext,
  registry: KernelRegistry,
  scalar: GPUBuffer,
  out: GPUBuffer,
  n: number,
): void {
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error(`dispatchBroadcastFillScalar: n must be positive (got ${String(n)})`);
  }
  const u = makeUniform(ctx, (v) => {
    v.setUint32(0, n, true);
    v.setUint32(4, 0, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
  });
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_BROADCAST_FILL_SCALAR),
    [bg(0, u), bg(1, scalar), bg(2, out)],
    n,
  );
  u.destroy();
}

/**
 * One fused AdamW step: updates **`theta`**, **`m`**, **`v`** in place; **`grad`** is read-only.
 * Host supplies bias corrections `1 - β₁ᵗ` and `1 - β₂ᵗ` in **`hyper.biasCorrection1/2`**.
 */
export function dispatchAdamWStep(
  ctx: GpuContext,
  registry: KernelRegistry,
  theta: GPUBuffer,
  grad: GPUBuffer,
  m: GPUBuffer,
  v: GPUBuffer,
  elementCount: number,
  hyper: {
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
    biasCorrection1: number;
    biasCorrection2: number;
  },
): void {
  const u = makeAdamUniform(ctx, elementCount, hyper);
  dispatchKernel(
    ctx,
    registry.get(CORE_KERNEL_ADAMW_STEP),
    [bg(0, u), bg(1, theta), bg(2, grad), bg(3, m), bg(4, v)],
    elementCount,
  );
  u.destroy();
}
