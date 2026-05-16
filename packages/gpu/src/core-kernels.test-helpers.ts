import {
  createBf16StorageBuffer,
  createFloat32StorageBuffer,
  float32ByteLength,
  readBf16Buffer,
  readFloat32Buffer,
  writeBf16Buffer,
  writeFloat32Buffer,
} from './buffer';
import { packBf16FromFloat32, unpackBf16ToFloat32 } from './bf16-pack';
import {
  dispatchAdd,
  dispatchAddBf16,
  dispatchGelu,
  dispatchGeluBackward,
  dispatchGeluBackwardBf16,
  dispatchGeluBf16,
  dispatchMatMul,
  dispatchMatMulBf16,
  dispatchMul,
  dispatchMulBf16,
  dispatchReduceMax,
  dispatchReduceMaxBf16,
  dispatchReduceSum,
  dispatchReduceSumBf16,
  dispatchRmsNorm,
  dispatchRmsNormBf16,
  dispatchRmsNormGradGammaBf16,
  dispatchRmsNormGradInputBf16,
  dispatchScale,
  dispatchScaleBf16,
  dispatchSoftmax,
  dispatchSoftmaxBackward,
  dispatchSoftmaxBackwardBf16,
  dispatchSoftmaxBf16,
  dispatchTranspose,
  dispatchTransposeBf16,
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

export async function smokeSoftmaxBackward(
  ctx: GpuContext,
  registry: KernelRegistry,
  gradOut: Float32Array,
  y: Float32Array,
  rows: number,
  cols: number,
): Promise<Float32Array> {
  const bufG = createFloat32StorageBuffer(ctx.device, rows * cols);
  const bufY = createFloat32StorageBuffer(ctx.device, rows * cols);
  const bufIn = createFloat32StorageBuffer(ctx.device, rows * cols);
  writeFloat32Buffer(ctx.queue, bufG, gradOut);
  writeFloat32Buffer(ctx.queue, bufY, y);
  return runAndRead(
    ctx,
    bufIn,
    rows * cols,
    () => {
      dispatchSoftmaxBackward(ctx, registry, bufG, bufY, bufIn, rows, cols);
    },
    [bufG, bufY, bufIn],
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

/** Test-only 16-bit smoke dtype — only `bf16` is supported (fp16 was dropped). */
export type Smoke16Dtype = 'bf16';

function writeBf16(queue: GPUQueue, buf: GPUBuffer, f32: Float32Array): void {
  writeBf16Buffer(queue, buf, packBf16FromFloat32(f32));
}

async function readBf16(
  device: GPUDevice,
  queue: GPUQueue,
  buf: GPUBuffer,
  n: number,
): Promise<Float32Array> {
  return unpackBf16ToFloat32(await readBf16Buffer(device, queue, buf, n));
}

export async function smokeMatMul16(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  b: Float32Array,
  M: number,
  N: number,
  K: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufA = createBf16StorageBuffer(ctx.device, M * K);
  const bufB = createBf16StorageBuffer(ctx.device, K * N);
  const bufC = createBf16StorageBuffer(ctx.device, M * N);
  writeBf16(ctx.queue, bufA, a);
  writeBf16(ctx.queue, bufB, b);
  dispatchMatMulBf16(ctx, registry, bufA, bufB, bufC, M, N, K);
  const out = await readBf16(ctx.device, ctx.queue, bufC, M * N);
  bufA.destroy();
  bufB.destroy();
  bufC.destroy();
  return out;
}

export async function smokeTranspose16(
  ctx: GpuContext,
  registry: KernelRegistry,
  src: Float32Array,
  rows: number,
  cols: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufIn = createBf16StorageBuffer(ctx.device, rows * cols);
  const bufOut = createBf16StorageBuffer(ctx.device, rows * cols);
  writeBf16(ctx.queue, bufIn, src);
  dispatchTransposeBf16(ctx, registry, bufIn, bufOut, rows, cols);
  const out = await readBf16(ctx.device, ctx.queue, bufOut, rows * cols);
  bufIn.destroy();
  bufOut.destroy();
  return out;
}

export async function smokeAdd16(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  b: Float32Array,
  n: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufA = createBf16StorageBuffer(ctx.device, n);
  const bufB = createBf16StorageBuffer(ctx.device, n);
  const bufO = createBf16StorageBuffer(ctx.device, n);
  writeBf16(ctx.queue, bufA, a);
  writeBf16(ctx.queue, bufB, b);
  dispatchAddBf16(ctx, registry, bufA, bufB, bufO, n);
  const out = await readBf16(ctx.device, ctx.queue, bufO, n);
  bufA.destroy();
  bufB.destroy();
  bufO.destroy();
  return out;
}

export async function smokeMul16(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  b: Float32Array,
  n: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufA = createBf16StorageBuffer(ctx.device, n);
  const bufB = createBf16StorageBuffer(ctx.device, n);
  const bufO = createBf16StorageBuffer(ctx.device, n);
  writeBf16(ctx.queue, bufA, a);
  writeBf16(ctx.queue, bufB, b);
  dispatchMulBf16(ctx, registry, bufA, bufB, bufO, n);
  const out = await readBf16(ctx.device, ctx.queue, bufO, n);
  bufA.destroy();
  bufB.destroy();
  bufO.destroy();
  return out;
}

export async function smokeScale16(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  n: number,
  scale: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufA = createBf16StorageBuffer(ctx.device, n);
  const bufO = createBf16StorageBuffer(ctx.device, n);
  writeBf16(ctx.queue, bufA, a);
  dispatchScaleBf16(ctx, registry, bufA, bufO, n, scale);
  const out = await readBf16(ctx.device, ctx.queue, bufO, n);
  bufA.destroy();
  bufO.destroy();
  return out;
}

export async function smokeGelu16(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  n: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufIn = createBf16StorageBuffer(ctx.device, n);
  const bufO = createBf16StorageBuffer(ctx.device, n);
  writeBf16(ctx.queue, bufIn, a);
  dispatchGeluBf16(ctx, registry, bufIn, bufO, n);
  const out = await readBf16(ctx.device, ctx.queue, bufO, n);
  bufIn.destroy();
  bufO.destroy();
  return out;
}

export async function smokeGeluBackward16(
  ctx: GpuContext,
  registry: KernelRegistry,
  gradOut: Float32Array,
  inp: Float32Array,
  n: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufG = createBf16StorageBuffer(ctx.device, n);
  const bufI = createBf16StorageBuffer(ctx.device, n);
  const bufO = createBf16StorageBuffer(ctx.device, n);
  writeBf16(ctx.queue, bufG, gradOut);
  writeBf16(ctx.queue, bufI, inp);
  dispatchGeluBackwardBf16(ctx, registry, bufG, bufI, bufO, n);
  const out = await readBf16(ctx.device, ctx.queue, bufO, n);
  bufG.destroy();
  bufI.destroy();
  bufO.destroy();
  return out;
}

export async function smokeSoftmaxBackward16(
  ctx: GpuContext,
  registry: KernelRegistry,
  gradOut: Float32Array,
  y: Float32Array,
  rows: number,
  cols: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const n = rows * cols;
  const bufG = createBf16StorageBuffer(ctx.device, n);
  const bufY = createBf16StorageBuffer(ctx.device, n);
  const bufO = createBf16StorageBuffer(ctx.device, n);
  writeBf16(ctx.queue, bufG, gradOut);
  writeBf16(ctx.queue, bufY, y);
  dispatchSoftmaxBackwardBf16(ctx, registry, bufG, bufY, bufO, rows, cols);
  const out = await readBf16(ctx.device, ctx.queue, bufO, n);
  bufG.destroy();
  bufY.destroy();
  bufO.destroy();
  return out;
}

export async function smokeRmsNormGradInput16(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  gamma: Float32Array,
  gradOut: Float32Array,
  rows: number,
  cols: number,
  _dtype: Smoke16Dtype = 'bf16',
  epsilon = 1e-6,
): Promise<Float32Array> {
  const bufIn = createBf16StorageBuffer(ctx.device, rows * cols);
  const bufGam = createBf16StorageBuffer(ctx.device, cols);
  const bufGo = createBf16StorageBuffer(ctx.device, rows * cols);
  const bufOut = createBf16StorageBuffer(ctx.device, rows * cols);
  writeBf16(ctx.queue, bufIn, data);
  writeBf16(ctx.queue, bufGam, gamma);
  writeBf16(ctx.queue, bufGo, gradOut);
  dispatchRmsNormGradInputBf16(
    ctx,
    registry,
    bufIn,
    bufGam,
    cols,
    bufGo,
    bufOut,
    rows,
    cols,
    epsilon,
  );
  const out = await readBf16(ctx.device, ctx.queue, bufOut, rows * cols);
  bufIn.destroy();
  bufGam.destroy();
  bufGo.destroy();
  bufOut.destroy();
  return out;
}

export async function smokeRmsNormGradGamma16(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  gradOut: Float32Array,
  rows: number,
  cols: number,
  _dtype: Smoke16Dtype = 'bf16',
  epsilon = 1e-6,
): Promise<Float32Array> {
  const bufIn = createBf16StorageBuffer(ctx.device, rows * cols);
  const bufGo = createBf16StorageBuffer(ctx.device, rows * cols);
  const bufOut = createBf16StorageBuffer(ctx.device, cols);
  writeBf16(ctx.queue, bufIn, data);
  writeBf16(ctx.queue, bufGo, gradOut);
  dispatchRmsNormGradGammaBf16(ctx, registry, bufIn, bufGo, bufOut, rows, cols, epsilon);
  const out = await readBf16(ctx.device, ctx.queue, bufOut, cols);
  bufIn.destroy();
  bufGo.destroy();
  bufOut.destroy();
  return out;
}

export async function smokeReduceSum16(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  rows: number,
  cols: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufIn = createBf16StorageBuffer(ctx.device, rows * cols);
  const bufOut = createBf16StorageBuffer(ctx.device, rows);
  writeBf16(ctx.queue, bufIn, data);
  dispatchReduceSumBf16(ctx, registry, bufIn, bufOut, rows, cols);
  const out = await readBf16(ctx.device, ctx.queue, bufOut, rows);
  bufIn.destroy();
  bufOut.destroy();
  return out;
}

export async function smokeReduceMax16(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  rows: number,
  cols: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufIn = createBf16StorageBuffer(ctx.device, rows * cols);
  const bufOut = createBf16StorageBuffer(ctx.device, rows);
  writeBf16(ctx.queue, bufIn, data);
  dispatchReduceMaxBf16(ctx, registry, bufIn, bufOut, rows, cols);
  const out = await readBf16(ctx.device, ctx.queue, bufOut, rows);
  bufIn.destroy();
  bufOut.destroy();
  return out;
}

export async function smokeSoftmax16(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  rows: number,
  cols: number,
  _dtype: Smoke16Dtype = 'bf16',
): Promise<Float32Array> {
  const bufIn = createBf16StorageBuffer(ctx.device, rows * cols);
  const bufOut = createBf16StorageBuffer(ctx.device, rows * cols);
  writeBf16(ctx.queue, bufIn, data);
  dispatchSoftmaxBf16(ctx, registry, bufIn, bufOut, rows, cols);
  const out = await readBf16(ctx.device, ctx.queue, bufOut, rows * cols);
  bufIn.destroy();
  bufOut.destroy();
  return out;
}

export async function smokeRmsNorm16(
  ctx: GpuContext,
  registry: KernelRegistry,
  data: Float32Array,
  gamma: Float32Array,
  rows: number,
  cols: number,
  _dtype: Smoke16Dtype = 'bf16',
  epsilon = 1e-6,
): Promise<Float32Array> {
  const bufIn = createBf16StorageBuffer(ctx.device, rows * cols);
  const bufG = createBf16StorageBuffer(ctx.device, gamma.length);
  const bufO = createBf16StorageBuffer(ctx.device, rows * cols);
  writeBf16(ctx.queue, bufIn, data);
  writeBf16(ctx.queue, bufG, gamma);
  dispatchRmsNormBf16(ctx, registry, bufIn, bufG, gamma.length, bufO, rows, cols, epsilon);
  const out = await readBf16(ctx.device, ctx.queue, bufO, rows * cols);
  bufIn.destroy();
  bufG.destroy();
  bufO.destroy();
  return out;
}
