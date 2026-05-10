import {
  createFloat32StorageBuffer,
  float32ByteLength,
  readFloat32Buffer,
  writeFloat32Buffer,
} from './buffer';
import {
  dispatchAdd,
  dispatchGelu,
  dispatchGeluBackward,
  dispatchMatMul,
  dispatchMul,
  dispatchReduceMax,
  dispatchReduceSum,
  dispatchRmsNorm,
  dispatchScale,
  dispatchSoftmax,
  dispatchTranspose,
} from './core-kernels';
import type { GpuContext } from './device';
import type { KernelRegistry } from './kernel-registry';

/**
 * End-to-end test conveniences: allocate inputs/output, upload, dispatch, read back,
 * and free GPU buffers. These are intentionally test-only — production callers
 * (tensor layer) should manage buffer lifetimes themselves.
 */

async function runAndRead(
  ctx: GpuContext,
  out: GPUBuffer,
  elementCount: number,
  dispatch: () => void,
  toFree: GPUBuffer[],
): Promise<Float32Array> {
  dispatch();
  const result = await readFloat32Buffer(
    ctx.device,
    ctx.queue,
    out,
    float32ByteLength(elementCount),
  );
  for (const b of toFree) b.destroy();
  return result;
}

export async function smokeMatMul(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  b: Float32Array,
  M: number,
  N: number,
  K: number,
): Promise<Float32Array> {
  const bufA = createFloat32StorageBuffer(ctx.device, M * K);
  const bufB = createFloat32StorageBuffer(ctx.device, K * N);
  const bufC = createFloat32StorageBuffer(ctx.device, M * N);
  writeFloat32Buffer(ctx.queue, bufA, a);
  writeFloat32Buffer(ctx.queue, bufB, b);
  return runAndRead(
    ctx,
    bufC,
    M * N,
    () => {
      dispatchMatMul(ctx, registry, bufA, bufB, bufC, M, N, K);
    },
    [bufA, bufB, bufC],
  );
}

export async function smokeTranspose(
  ctx: GpuContext,
  registry: KernelRegistry,
  src: Float32Array,
  rows: number,
  cols: number,
): Promise<Float32Array> {
  const bufIn = createFloat32StorageBuffer(ctx.device, rows * cols);
  const bufOut = createFloat32StorageBuffer(ctx.device, rows * cols);
  writeFloat32Buffer(ctx.queue, bufIn, src);
  return runAndRead(
    ctx,
    bufOut,
    rows * cols,
    () => {
      dispatchTranspose(ctx, registry, bufIn, bufOut, rows, cols);
    },
    [bufIn, bufOut],
  );
}

export async function smokeAdd(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  b: Float32Array,
  n: number,
): Promise<Float32Array> {
  const bufA = createFloat32StorageBuffer(ctx.device, n);
  const bufB = createFloat32StorageBuffer(ctx.device, n);
  const bufOut = createFloat32StorageBuffer(ctx.device, n);
  writeFloat32Buffer(ctx.queue, bufA, a);
  writeFloat32Buffer(ctx.queue, bufB, b);
  return runAndRead(
    ctx,
    bufOut,
    n,
    () => {
      dispatchAdd(ctx, registry, bufA, bufB, bufOut, n);
    },
    [bufA, bufB, bufOut],
  );
}

export async function smokeMul(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  b: Float32Array,
  n: number,
): Promise<Float32Array> {
  const bufA = createFloat32StorageBuffer(ctx.device, n);
  const bufB = createFloat32StorageBuffer(ctx.device, n);
  const bufOut = createFloat32StorageBuffer(ctx.device, n);
  writeFloat32Buffer(ctx.queue, bufA, a);
  writeFloat32Buffer(ctx.queue, bufB, b);
  return runAndRead(
    ctx,
    bufOut,
    n,
    () => {
      dispatchMul(ctx, registry, bufA, bufB, bufOut, n);
    },
    [bufA, bufB, bufOut],
  );
}

export async function smokeScale(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  n: number,
  scale: number,
): Promise<Float32Array> {
  const bufA = createFloat32StorageBuffer(ctx.device, n);
  const bufOut = createFloat32StorageBuffer(ctx.device, n);
  writeFloat32Buffer(ctx.queue, bufA, a);
  return runAndRead(
    ctx,
    bufOut,
    n,
    () => {
      dispatchScale(ctx, registry, bufA, bufOut, n, scale);
    },
    [bufA, bufOut],
  );
}

export async function smokeGelu(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  n: number,
): Promise<Float32Array> {
  const bufIn = createFloat32StorageBuffer(ctx.device, n);
  const bufOut = createFloat32StorageBuffer(ctx.device, n);
  writeFloat32Buffer(ctx.queue, bufIn, a);
  return runAndRead(
    ctx,
    bufOut,
    n,
    () => {
      dispatchGelu(ctx, registry, bufIn, bufOut, n);
    },
    [bufIn, bufOut],
  );
}

export async function smokeGeluBackward(
  ctx: GpuContext,
  registry: KernelRegistry,
  gradOut: Float32Array,
  inp: Float32Array,
  n: number,
): Promise<Float32Array> {
  const bufGradOut = createFloat32StorageBuffer(ctx.device, n);
  const bufInp = createFloat32StorageBuffer(ctx.device, n);
  const bufGradIn = createFloat32StorageBuffer(ctx.device, n);
  writeFloat32Buffer(ctx.queue, bufGradOut, gradOut);
  writeFloat32Buffer(ctx.queue, bufInp, inp);
  return runAndRead(
    ctx,
    bufGradIn,
    n,
    () => {
      dispatchGeluBackward(ctx, registry, bufGradOut, bufInp, bufGradIn, n);
    },
    [bufGradOut, bufInp, bufGradIn],
  );
}

async function smokeRowwiseReduce(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  rows: number,
  cols: number,
  dispatch: (input: GPUBuffer, out: GPUBuffer) => void,
): Promise<Float32Array> {
  const bufIn = createFloat32StorageBuffer(ctx.device, rows * cols);
  const bufOut = createFloat32StorageBuffer(ctx.device, rows);
  writeFloat32Buffer(ctx.queue, bufIn, data);
  return runAndRead(
    ctx,
    bufOut,
    rows,
    () => {
      dispatch(bufIn, bufOut);
    },
    [bufIn, bufOut],
  );
}

export async function smokeReduceSum(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  rows: number,
  cols: number,
): Promise<Float32Array> {
  return smokeRowwiseReduce(ctx, registry, data, rows, cols, (input, out) => {
    dispatchReduceSum(ctx, registry, input, out, rows, cols);
  });
}

export async function smokeReduceMax(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  rows: number,
  cols: number,
): Promise<Float32Array> {
  return smokeRowwiseReduce(ctx, registry, data, rows, cols, (input, out) => {
    dispatchReduceMax(ctx, registry, input, out, rows, cols);
  });
}

export async function smokeSoftmax(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  rows: number,
  cols: number,
): Promise<Float32Array> {
  const bufIn = createFloat32StorageBuffer(ctx.device, rows * cols);
  const bufOut = createFloat32StorageBuffer(ctx.device, rows * cols);
  writeFloat32Buffer(ctx.queue, bufIn, data);
  return runAndRead(
    ctx,
    bufOut,
    rows * cols,
    () => {
      dispatchSoftmax(ctx, registry, bufIn, bufOut, rows, cols);
    },
    [bufIn, bufOut],
  );
}

export async function smokeRmsNorm(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  gamma: Float32Array,
  rows: number,
  cols: number,
  epsilon = 1e-6,
): Promise<Float32Array> {
  const bufIn = createFloat32StorageBuffer(ctx.device, rows * cols);
  const bufGamma = createFloat32StorageBuffer(ctx.device, gamma.length);
  const bufOut = createFloat32StorageBuffer(ctx.device, rows * cols);
  writeFloat32Buffer(ctx.queue, bufIn, data);
  writeFloat32Buffer(ctx.queue, bufGamma, gamma);
  return runAndRead(
    ctx,
    bufOut,
    rows * cols,
    () => {
      dispatchRmsNorm(ctx, registry, bufIn, bufGamma, gamma.length, bufOut, rows, cols, epsilon);
    },
    [bufIn, bufGamma, bufOut],
  );
}
