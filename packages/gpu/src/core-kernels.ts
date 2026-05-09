import type { GpuContext } from './device';
import type { KernelRegistry } from './kernel-registry';
import {
  loadElementwiseAddWgsl,
  loadElementwiseGeluWgsl,
  loadElementwiseMulWgsl,
  loadElementwiseScaleWgsl,
  loadMatMulWgsl,
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

/** All core numeric kernels registered by {@link registerCoreKernels}. */
export const CORE_KERNEL_IDS = [
  CORE_KERNEL_MATMUL,
  CORE_KERNEL_TRANSPOSE,
  CORE_KERNEL_ADD,
  CORE_KERNEL_MUL,
  CORE_KERNEL_SCALE,
  CORE_KERNEL_GELU,
] as const;

const WGSL_MAIN = 'main';
const WORKGROUP_SIZE = 256;
/** All core kernel uniforms fit in a 16-byte block (vec4-sized). */
const UNIFORM_BYTES = 16;

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
