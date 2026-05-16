import {
  createBf16StorageBuffer,
  createFloat32StorageBuffer,
  createFp16StorageBuffer,
  float32ByteLength,
  readBf16Buffer,
  readFloat32Buffer,
  readFp16Buffer,
  writeBf16Buffer,
  writeFloat32Buffer,
  writeFp16Buffer,
} from './buffer';
import {
  unpackBf16ToFloat32,
  unpackFp16ToFloat32,
  packBf16FromFloat32,
  packFp16FromFloat32,
} from './bf16-fp16-pack';
import {
  dispatchAdd,
  dispatchAddBf16,
  dispatchAddFp16,
  dispatchBroadcastFillScalarBf16,
  dispatchBroadcastFillScalarFp16,
  dispatchCrossEntropyFromLogitsBf16,
  dispatchCrossEntropyFromLogitsFp16,
  dispatchEmbeddingGatherBf16,
  dispatchEmbeddingGatherFp16,
  dispatchGelu,
  dispatchGeluBackward,
  dispatchGeluBackwardBf16,
  dispatchGeluBackwardFp16,
  dispatchGeluBf16,
  dispatchGeluFp16,
  dispatchMatMul,
  dispatchMatMulBf16,
  dispatchMatMulFp16,
  dispatchMul,
  dispatchMulBf16,
  dispatchMulFp16,
  dispatchReduceMax,
  dispatchReduceMaxBf16,
  dispatchReduceMaxFp16,
  dispatchReduceSum,
  dispatchReduceSumBf16,
  dispatchReduceSumFp16,
  dispatchRmsNorm,
  dispatchRmsNormBf16,
  dispatchRmsNormFp16,
  dispatchRmsNormGradGamma,
  dispatchRmsNormGradGammaBf16,
  dispatchRmsNormGradGammaFp16,
  dispatchRmsNormGradInput,
  dispatchRmsNormGradInputBf16,
  dispatchRmsNormGradInputFp16,
  dispatchScale,
  dispatchScaleBf16,
  dispatchScaleFp16,
  dispatchSoftmax,
  dispatchSoftmaxBackward,
  dispatchSoftmaxBackwardBf16,
  dispatchSoftmaxBackwardFp16,
  dispatchSoftmaxBf16,
  dispatchSoftmaxFp16,
  dispatchTranspose,
  dispatchTransposeBf16,
  dispatchTransposeFp16,
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

/** Test-only 16-bit smoke dtype (`bf16` **u32** pack or **`fp16`** packed/native per registry). */
export type Smoke16Dtype = 'bf16' | 'fp16';

function create16(device: GPUDevice, n: number, d: Smoke16Dtype): GPUBuffer {
  return d === 'bf16' ? createBf16StorageBuffer(device, n) : createFp16StorageBuffer(device, n);
}

function write16(queue: GPUQueue, buf: GPUBuffer, f32: Float32Array, d: Smoke16Dtype): void {
  const p = d === 'bf16' ? packBf16FromFloat32(f32) : packFp16FromFloat32(f32);
  if (d === 'bf16') {
    writeBf16Buffer(queue, buf, p);
  } else {
    writeFp16Buffer(queue, buf, p);
  }
}

async function read16(
  device: GPUDevice,
  queue: GPUQueue,
  buf: GPUBuffer,
  n: number,
  d: Smoke16Dtype,
): Promise<Float32Array> {
  if (d === 'bf16') {
    return unpackBf16ToFloat32(await readBf16Buffer(device, queue, buf, n));
  }
  return unpackFp16ToFloat32(await readFp16Buffer(device, queue, buf, n));
}

export async function smokeMatMul16(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  b: Float32Array,
  M: number,
  N: number,
  K: number,
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufA = create16(ctx.device, M * K, dtype);
  const bufB = create16(ctx.device, K * N, dtype);
  const bufC = create16(ctx.device, M * N, dtype);
  write16(ctx.queue, bufA, a, dtype);
  write16(ctx.queue, bufB, b, dtype);
  if (dtype === 'bf16') {
    dispatchMatMulBf16(ctx, registry, bufA, bufB, bufC, M, N, K);
  } else {
    dispatchMatMulFp16(ctx, registry, bufA, bufB, bufC, M, N, K);
  }
  const out = await read16(ctx.device, ctx.queue, bufC, M * N, dtype);
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
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufIn = create16(ctx.device, rows * cols, dtype);
  const bufOut = create16(ctx.device, rows * cols, dtype);
  write16(ctx.queue, bufIn, src, dtype);
  if (dtype === 'bf16') {
    dispatchTransposeBf16(ctx, registry, bufIn, bufOut, rows, cols);
  } else {
    dispatchTransposeFp16(ctx, registry, bufIn, bufOut, rows, cols);
  }
  const out = await read16(ctx.device, ctx.queue, bufOut, rows * cols, dtype);
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
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufA = create16(ctx.device, n, dtype);
  const bufB = create16(ctx.device, n, dtype);
  const bufO = create16(ctx.device, n, dtype);
  write16(ctx.queue, bufA, a, dtype);
  write16(ctx.queue, bufB, b, dtype);
  if (dtype === 'bf16') {
    dispatchAddBf16(ctx, registry, bufA, bufB, bufO, n);
  } else {
    dispatchAddFp16(ctx, registry, bufA, bufB, bufO, n);
  }
  const out = await read16(ctx.device, ctx.queue, bufO, n, dtype);
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
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufA = create16(ctx.device, n, dtype);
  const bufB = create16(ctx.device, n, dtype);
  const bufO = create16(ctx.device, n, dtype);
  write16(ctx.queue, bufA, a, dtype);
  write16(ctx.queue, bufB, b, dtype);
  if (dtype === 'bf16') {
    dispatchMulBf16(ctx, registry, bufA, bufB, bufO, n);
  } else {
    dispatchMulFp16(ctx, registry, bufA, bufB, bufO, n);
  }
  const out = await read16(ctx.device, ctx.queue, bufO, n, dtype);
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
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufA = create16(ctx.device, n, dtype);
  const bufO = create16(ctx.device, n, dtype);
  write16(ctx.queue, bufA, a, dtype);
  if (dtype === 'bf16') {
    dispatchScaleBf16(ctx, registry, bufA, bufO, n, scale);
  } else {
    dispatchScaleFp16(ctx, registry, bufA, bufO, n, scale);
  }
  const out = await read16(ctx.device, ctx.queue, bufO, n, dtype);
  bufA.destroy();
  bufO.destroy();
  return out;
}

export async function smokeGelu16(
  ctx: GpuContext,
  registry: KernelRegistry,
  a: Float32Array,
  n: number,
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufIn = create16(ctx.device, n, dtype);
  const bufO = create16(ctx.device, n, dtype);
  write16(ctx.queue, bufIn, a, dtype);
  if (dtype === 'bf16') {
    dispatchGeluBf16(ctx, registry, bufIn, bufO, n);
  } else {
    dispatchGeluFp16(ctx, registry, bufIn, bufO, n);
  }
  const out = await read16(ctx.device, ctx.queue, bufO, n, dtype);
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
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufG = create16(ctx.device, n, dtype);
  const bufI = create16(ctx.device, n, dtype);
  const bufO = create16(ctx.device, n, dtype);
  write16(ctx.queue, bufG, gradOut, dtype);
  write16(ctx.queue, bufI, inp, dtype);
  if (dtype === 'bf16') {
    dispatchGeluBackwardBf16(ctx, registry, bufG, bufI, bufO, n);
  } else {
    dispatchGeluBackwardFp16(ctx, registry, bufG, bufI, bufO, n);
  }
  const out = await read16(ctx.device, ctx.queue, bufO, n, dtype);
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
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const n = rows * cols;
  const bufG = create16(ctx.device, n, dtype);
  const bufY = create16(ctx.device, n, dtype);
  const bufO = create16(ctx.device, n, dtype);
  write16(ctx.queue, bufG, gradOut, dtype);
  write16(ctx.queue, bufY, y, dtype);
  if (dtype === 'bf16') {
    dispatchSoftmaxBackwardBf16(ctx, registry, bufG, bufY, bufO, rows, cols);
  } else {
    dispatchSoftmaxBackwardFp16(ctx, registry, bufG, bufY, bufO, rows, cols);
  }
  const out = await read16(ctx.device, ctx.queue, bufO, n, dtype);
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
  dtype: Smoke16Dtype,
  epsilon = 1e-6,
): Promise<Float32Array> {
  const bufIn = create16(ctx.device, rows * cols, dtype);
  const bufGam = create16(ctx.device, cols, dtype);
  const bufGo = create16(ctx.device, rows * cols, dtype);
  const bufOut = create16(ctx.device, rows * cols, dtype);
  write16(ctx.queue, bufIn, data, dtype);
  write16(ctx.queue, bufGam, gamma, dtype);
  write16(ctx.queue, bufGo, gradOut, dtype);
  if (dtype === 'bf16') {
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
  } else {
    dispatchRmsNormGradInputFp16(
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
  }
  const out = await read16(ctx.device, ctx.queue, bufOut, rows * cols, dtype);
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
  dtype: Smoke16Dtype,
  epsilon = 1e-6,
): Promise<Float32Array> {
  const bufIn = create16(ctx.device, rows * cols, dtype);
  const bufGo = create16(ctx.device, rows * cols, dtype);
  const bufOut = create16(ctx.device, cols, dtype);
  write16(ctx.queue, bufIn, data, dtype);
  write16(ctx.queue, bufGo, gradOut, dtype);
  if (dtype === 'bf16') {
    dispatchRmsNormGradGammaBf16(ctx, registry, bufIn, bufGo, bufOut, rows, cols, epsilon);
  } else {
    dispatchRmsNormGradGammaFp16(ctx, registry, bufIn, bufGo, bufOut, rows, cols, epsilon);
  }
  const out = await read16(ctx.device, ctx.queue, bufOut, cols, dtype);
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
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufIn = create16(ctx.device, rows * cols, dtype);
  const bufOut = create16(ctx.device, rows, dtype);
  write16(ctx.queue, bufIn, data, dtype);
  if (dtype === 'bf16') {
    dispatchReduceSumBf16(ctx, registry, bufIn, bufOut, rows, cols);
  } else {
    dispatchReduceSumFp16(ctx, registry, bufIn, bufOut, rows, cols);
  }
  const out = await read16(ctx.device, ctx.queue, bufOut, rows, dtype);
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
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufIn = create16(ctx.device, rows * cols, dtype);
  const bufOut = create16(ctx.device, rows, dtype);
  write16(ctx.queue, bufIn, data, dtype);
  if (dtype === 'bf16') {
    dispatchReduceMaxBf16(ctx, registry, bufIn, bufOut, rows, cols);
  } else {
    dispatchReduceMaxFp16(ctx, registry, bufIn, bufOut, rows, cols);
  }
  const out = await read16(ctx.device, ctx.queue, bufOut, rows, dtype);
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
  dtype: Smoke16Dtype,
): Promise<Float32Array> {
  const bufIn = create16(ctx.device, rows * cols, dtype);
  const bufOut = create16(ctx.device, rows * cols, dtype);
  write16(ctx.queue, bufIn, data, dtype);
  if (dtype === 'bf16') {
    dispatchSoftmaxBf16(ctx, registry, bufIn, bufOut, rows, cols);
  } else {
    dispatchSoftmaxFp16(ctx, registry, bufIn, bufOut, rows, cols);
  }
  const out = await read16(ctx.device, ctx.queue, bufOut, rows * cols, dtype);
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
  dtype: Smoke16Dtype,
  epsilon = 1e-6,
): Promise<Float32Array> {
  const bufIn = create16(ctx.device, rows * cols, dtype);
  const bufG = create16(ctx.device, gamma.length, dtype);
  const bufO = create16(ctx.device, rows * cols, dtype);
  write16(ctx.queue, bufIn, data, dtype);
  write16(ctx.queue, bufG, gamma, dtype);
  if (dtype === 'bf16') {
    dispatchRmsNormBf16(ctx, registry, bufIn, bufG, gamma.length, bufO, rows, cols, epsilon);
  } else {
    dispatchRmsNormFp16(ctx, registry, bufIn, bufG, gamma.length, bufO, rows, cols, epsilon);
  }
  const out = await read16(ctx.device, ctx.queue, bufO, rows * cols, dtype);
  bufIn.destroy();
  bufG.destroy();
  bufO.destroy();
  return out;
}
