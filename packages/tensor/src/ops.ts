import {
  dispatchAdd,
  dispatchAddBf16,
  dispatchAdamWStep,
  dispatchBroadcastFillScalar,
  dispatchBroadcastFillScalarBf16,
  dispatchCrossEntropyFromLogits,
  dispatchCrossEntropyFromLogitsBf16,
  dispatchCrossEntropyLogitsGrad,
  dispatchCrossEntropyLogitsGradBf16,
  dispatchEmbeddingGather,
  dispatchEmbeddingGatherBf16,
  dispatchEmbeddingScatterAccum,
  dispatchEmbeddingScatterAccumBf16,
  dispatchGelu,
  dispatchGeluBackward,
  dispatchGeluBackwardBf16,
  dispatchGeluBf16,
  dispatchMatMul,
  dispatchMatMulBf16,
  dispatchMul,
  dispatchMulBf16,
  dispatchReduceSum,
  dispatchReduceSumBf16,
  dispatchRmsNorm,
  dispatchRmsNormBf16,
  dispatchRmsNormGradGamma,
  dispatchRmsNormGradGammaBf16,
  dispatchRmsNormGradInput,
  dispatchRmsNormGradInputBf16,
  dispatchScale,
  dispatchScaleBf16,
  dispatchSoftmax,
  dispatchSoftmaxBackward,
  dispatchSoftmaxBackwardBf16,
  dispatchSoftmaxBf16,
  dispatchTranspose,
  dispatchTransposeBf16,
} from '@llm-transformer-diy/gpu';

import {
  assertDtype,
  assertFloatDtype,
  assertRank,
  assertSameDtype,
  assertSameShape,
  computeContiguousStrides,
  floatDtypeOf,
  isContiguous,
  productOfShape,
  type FloatDType,
} from './shape';
import { allocateContiguousBuffer, createZeros, makeContiguousTensor } from './tensor';
import type { Tensor, TensorRuntime } from './types';

function dispatchMatMulD(
  rt: TensorRuntime,
  a: GPUBuffer,
  b: GPUBuffer,
  out: GPUBuffer,
  M: number,
  N: number,
  K: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchMatMul(ctx, registry, a, b, out, M, N, K);
  } else {
    dispatchMatMulBf16(ctx, registry, a, b, out, M, N, K);
  }
}

function dispatchTransposeD(
  rt: TensorRuntime,
  src: GPUBuffer,
  dst: GPUBuffer,
  rows: number,
  cols: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchTranspose(ctx, registry, src, dst, rows, cols);
  } else {
    dispatchTransposeBf16(ctx, registry, src, dst, rows, cols);
  }
}

function dispatchAddD(
  rt: TensorRuntime,
  a: GPUBuffer,
  b: GPUBuffer,
  out: GPUBuffer,
  n: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchAdd(ctx, registry, a, b, out, n);
  } else {
    dispatchAddBf16(ctx, registry, a, b, out, n);
  }
}

function dispatchMulD(
  rt: TensorRuntime,
  a: GPUBuffer,
  b: GPUBuffer,
  out: GPUBuffer,
  n: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchMul(ctx, registry, a, b, out, n);
  } else {
    dispatchMulBf16(ctx, registry, a, b, out, n);
  }
}

function dispatchScaleD(
  rt: TensorRuntime,
  inp: GPUBuffer,
  out: GPUBuffer,
  n: number,
  k: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchScale(ctx, registry, inp, out, n, k);
  } else {
    dispatchScaleBf16(ctx, registry, inp, out, n, k);
  }
}

function dispatchGeluD(
  rt: TensorRuntime,
  inp: GPUBuffer,
  out: GPUBuffer,
  n: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchGelu(ctx, registry, inp, out, n);
  } else {
    dispatchGeluBf16(ctx, registry, inp, out, n);
  }
}

function dispatchSoftmaxD(
  rt: TensorRuntime,
  inp: GPUBuffer,
  out: GPUBuffer,
  rows: number,
  cols: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchSoftmax(ctx, registry, inp, out, rows, cols);
  } else {
    dispatchSoftmaxBf16(ctx, registry, inp, out, rows, cols);
  }
}

function dispatchReduceSumD(
  rt: TensorRuntime,
  inp: GPUBuffer,
  out: GPUBuffer,
  rows: number,
  cols: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchReduceSum(ctx, registry, inp, out, rows, cols);
  } else {
    dispatchReduceSumBf16(ctx, registry, inp, out, rows, cols);
  }
}

function dispatchRmsNormD(
  rt: TensorRuntime,
  x: GPUBuffer,
  gamma: GPUBuffer,
  gammaLen: number,
  out: GPUBuffer,
  rows: number,
  cols: number,
  epsilon: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchRmsNorm(ctx, registry, x, gamma, gammaLen, out, rows, cols, epsilon);
  } else {
    dispatchRmsNormBf16(ctx, registry, x, gamma, gammaLen, out, rows, cols, epsilon);
  }
}

function dispatchGeluBackwardD(
  rt: TensorRuntime,
  gradOut: GPUBuffer,
  inp: GPUBuffer,
  gradIn: GPUBuffer,
  n: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchGeluBackward(ctx, registry, gradOut, inp, gradIn, n);
  } else {
    dispatchGeluBackwardBf16(ctx, registry, gradOut, inp, gradIn, n);
  }
}

function dispatchSoftmaxBackwardD(
  rt: TensorRuntime,
  gradOut: GPUBuffer,
  y: GPUBuffer,
  gradIn: GPUBuffer,
  rows: number,
  cols: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchSoftmaxBackward(ctx, registry, gradOut, y, gradIn, rows, cols);
  } else {
    dispatchSoftmaxBackwardBf16(ctx, registry, gradOut, y, gradIn, rows, cols);
  }
}

function dispatchRmsNormGradInputD(
  rt: TensorRuntime,
  x: GPUBuffer,
  gamma: GPUBuffer,
  gammaLen: number,
  gradOut: GPUBuffer,
  gradIn: GPUBuffer,
  rows: number,
  cols: number,
  epsilon: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchRmsNormGradInput(
      ctx,
      registry,
      x,
      gamma,
      gammaLen,
      gradOut,
      gradIn,
      rows,
      cols,
      epsilon,
    );
  } else {
    dispatchRmsNormGradInputBf16(
      ctx,
      registry,
      x,
      gamma,
      gammaLen,
      gradOut,
      gradIn,
      rows,
      cols,
      epsilon,
    );
  }
}

function dispatchRmsNormGradGammaD(
  rt: TensorRuntime,
  x: GPUBuffer,
  gradOut: GPUBuffer,
  gradGamma: GPUBuffer,
  rows: number,
  cols: number,
  epsilon: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchRmsNormGradGamma(ctx, registry, x, gradOut, gradGamma, rows, cols, epsilon);
  } else {
    dispatchRmsNormGradGammaBf16(ctx, registry, x, gradOut, gradGamma, rows, cols, epsilon);
  }
}

function dispatchEmbeddingScatterAccumD(
  rt: TensorRuntime,
  gradTokens: GPUBuffer,
  indices: GPUBuffer,
  gradWeights: GPUBuffer,
  numTokens: number,
  embedDim: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchEmbeddingScatterAccum(
      ctx,
      registry,
      gradTokens,
      indices,
      gradWeights,
      numTokens,
      embedDim,
    );
  } else {
    dispatchEmbeddingScatterAccumBf16(
      ctx,
      registry,
      gradTokens,
      indices,
      gradWeights,
      numTokens,
      embedDim,
    );
  }
}

function dispatchCrossEntropyLogitsGradD(
  rt: TensorRuntime,
  probs: GPUBuffer,
  targets: GPUBuffer,
  upstreamPerRow: GPUBuffer,
  gradLogitsOut: GPUBuffer,
  rows: number,
  cols: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchCrossEntropyLogitsGrad(
      ctx,
      registry,
      probs,
      targets,
      upstreamPerRow,
      gradLogitsOut,
      rows,
      cols,
    );
  } else {
    dispatchCrossEntropyLogitsGradBf16(
      ctx,
      registry,
      probs,
      targets,
      upstreamPerRow,
      gradLogitsOut,
      rows,
      cols,
    );
  }
}

function dispatchEmbeddingGatherD(
  rt: TensorRuntime,
  weights: GPUBuffer,
  indices: GPUBuffer,
  out: GPUBuffer,
  tokens: number,
  embedDim: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchEmbeddingGather(ctx, registry, weights, indices, out, tokens, embedDim);
  } else {
    dispatchEmbeddingGatherBf16(ctx, registry, weights, indices, out, tokens, embedDim);
  }
}

function dispatchCrossEntropyFromLogitsD(
  rt: TensorRuntime,
  logits: GPUBuffer,
  targets: GPUBuffer,
  out: GPUBuffer,
  rows: number,
  cols: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchCrossEntropyFromLogits(ctx, registry, logits, targets, out, rows, cols);
  } else {
    dispatchCrossEntropyFromLogitsBf16(ctx, registry, logits, targets, out, rows, cols);
  }
}

function dispatchBroadcastFillScalarD(
  rt: TensorRuntime,
  scalar: GPUBuffer,
  out: GPUBuffer,
  n: number,
  dt: FloatDType,
): void {
  const { ctx, registry } = rt;
  if (dt === 'f32') {
    dispatchBroadcastFillScalar(ctx, registry, scalar, out, n);
  } else {
    dispatchBroadcastFillScalarBf16(ctx, registry, scalar, out, n);
  }
}

/**
 * Matrix multiply for **rank-2** row-major tensors: `a` is `[M, K]`, `b` is
 * `[K, N]`, output is `[M, N]`. Throws on rank or inner-dim mismatch.
 */
export function matmul(rt: TensorRuntime, a: Tensor, b: Tensor): Tensor {
  assertRank(a, 2);
  assertRank(b, 2);
  assertFloatDtype(a);
  assertFloatDtype(b);
  assertSameDtype(a, b);
  const dt = floatDtypeOf(a);
  const M = a.shape[0];
  const K = a.shape[1];
  const N = b.shape[1];
  if (b.shape[0] !== K) {
    throw new Error(
      `matmul: inner dimensions must match (a=[${a.shape.join(',')}], b=[${b.shape.join(',')}])`,
    );
  }
  const out = allocateContiguousBuffer(rt, M * N, dt);
  dispatchMatMulD(rt, a.buffer, b.buffer, out, M, N, K, dt);
  return makeContiguousTensor([M, N], out, dt);
}

/** Elementwise `out[i] = a[i] + b[i]`. Requires equal shapes (no broadcasting). */
export function add(rt: TensorRuntime, a: Tensor, b: Tensor): Tensor {
  assertSameShape(a, b);
  assertFloatDtype(a);
  assertFloatDtype(b);
  assertSameDtype(a, b);
  const dt = floatDtypeOf(a);
  const out = allocateContiguousBuffer(rt, a.length, dt);
  dispatchAddD(rt, a.buffer, b.buffer, out, a.length, dt);
  return makeContiguousTensor(a.shape, out, dt);
}

/** Elementwise `out[i] = a[i] * b[i]`. Requires equal shapes (no broadcasting). */
export function mul(rt: TensorRuntime, a: Tensor, b: Tensor): Tensor {
  assertSameShape(a, b);
  assertFloatDtype(a);
  assertFloatDtype(b);
  assertSameDtype(a, b);
  const dt = floatDtypeOf(a);
  const out = allocateContiguousBuffer(rt, a.length, dt);
  dispatchMulD(rt, a.buffer, b.buffer, out, a.length, dt);
  return makeContiguousTensor(a.shape, out, dt);
}

/** Elementwise `out[i] = a[i] * k`. */
export function scale(rt: TensorRuntime, a: Tensor, k: number): Tensor {
  assertFloatDtype(a);
  const dt = floatDtypeOf(a);
  const out = allocateContiguousBuffer(rt, a.length, dt);
  dispatchScaleD(rt, a.buffer, out, a.length, k, dt);
  return makeContiguousTensor(a.shape, out, dt);
}

/**
 * Row-wise softmax for **rank-2** row-major `[rows, cols]` (numerically stable;
 * matches `core.softmax`).
 */
export function softmax(rt: TensorRuntime, a: Tensor): Tensor {
  assertRank(a, 2);
  assertFloatDtype(a);
  if (!isContiguous(a)) {
    throw new Error('softmax: non-contiguous strides not supported');
  }
  const dt = floatDtypeOf(a);
  const rows = a.shape[0];
  const cols = a.shape[1];
  const out = allocateContiguousBuffer(rt, rows * cols, dt);
  dispatchSoftmaxD(rt, a.buffer, out, rows, cols, dt);
  return makeContiguousTensor([rows, cols], out, dt);
}

/**
 * Softmax backward: `gradOut` and `softmaxOutput` (`y`) must match shapes
 * `[rows, cols]` from the forward softmax.
 */
export function softmaxBackward(
  rt: TensorRuntime,
  gradOutput: Tensor,
  softmaxOutput: Tensor,
): Tensor {
  assertRank(gradOutput, 2);
  assertRank(softmaxOutput, 2);
  assertSameShape(gradOutput, softmaxOutput);
  assertFloatDtype(gradOutput);
  assertFloatDtype(softmaxOutput);
  assertSameDtype(gradOutput, softmaxOutput);
  if (!isContiguous(gradOutput) || !isContiguous(softmaxOutput)) {
    throw new Error('softmaxBackward: non-contiguous strides not supported');
  }
  const dt = floatDtypeOf(gradOutput);
  const rows = gradOutput.shape[0];
  const cols = gradOutput.shape[1];
  const out = allocateContiguousBuffer(rt, rows * cols, dt);
  dispatchSoftmaxBackwardD(rt, gradOutput.buffer, softmaxOutput.buffer, out, rows, cols, dt);
  return makeContiguousTensor([rows, cols], out, dt);
}

/** Elementwise GELU activation (tanh formulation; matches `geluScalar` in `@llm-transformer-diy/gpu`). */
export function gelu(rt: TensorRuntime, a: Tensor): Tensor {
  assertFloatDtype(a);
  const dt = floatDtypeOf(a);
  const out = allocateContiguousBuffer(rt, a.length, dt);
  dispatchGeluD(rt, a.buffer, out, a.length, dt);
  return makeContiguousTensor(a.shape, out, dt);
}

/**
 * GELU backward: `grad_in[i] = grad_out[i] * gelu′(input[i])`. Shapes must
 * match elementwise (no broadcasting).
 */
export function geluBackward(rt: TensorRuntime, gradOutput: Tensor, input: Tensor): Tensor {
  assertSameShape(gradOutput, input);
  assertFloatDtype(gradOutput);
  assertFloatDtype(input);
  assertSameDtype(gradOutput, input);
  const dt = floatDtypeOf(input);
  const out = allocateContiguousBuffer(rt, input.length, dt);
  dispatchGeluBackwardD(rt, gradOutput.buffer, input.buffer, out, input.length, dt);
  return makeContiguousTensor(input.shape, out, dt);
}

/** Row-wise **RMSNorm** on rank-2 **`[rows, cols]`** with **`gamma: [cols]`**. */
export function rmsnorm(rt: TensorRuntime, x: Tensor, gamma: Tensor, epsilon = 1e-6): Tensor {
  assertFloatDtype(x);
  assertFloatDtype(gamma);
  assertSameDtype(x, gamma);
  assertRank(x, 2);
  assertRank(gamma, 1);
  const dt = floatDtypeOf(x);
  const rows = x.shape[0];
  const cols = x.shape[1];
  if (gamma.shape[0] !== cols) {
    throw new Error(`rmsnorm: gamma length ${String(gamma.shape[0])} != d_model ${String(cols)}`);
  }
  if (!isContiguous(x) || !isContiguous(gamma)) {
    throw new Error('rmsnorm: non-contiguous tensors not supported');
  }
  const out = allocateContiguousBuffer(rt, rows * cols, dt);
  dispatchRmsNormD(rt, x.buffer, gamma.buffer, cols, out, rows, cols, epsilon, dt);
  return makeContiguousTensor([rows, cols], out, dt);
}

/** ∂L/∂x for {@link rmsnorm}. */
export function rmsnormGradInput(
  rt: TensorRuntime,
  gradOutput: Tensor,
  x: Tensor,
  gamma: Tensor,
  epsilon = 1e-6,
): Tensor {
  assertSameShape(gradOutput, x);
  assertFloatDtype(gradOutput);
  assertFloatDtype(x);
  assertFloatDtype(gamma);
  assertSameDtype(gradOutput, x);
  assertSameDtype(x, gamma);
  assertRank(x, 2);
  assertRank(gamma, 1);
  const rows = x.shape[0];
  const cols = x.shape[1];
  const dt = floatDtypeOf(x);
  if (
    gamma.shape[0] !== cols ||
    !isContiguous(x) ||
    !isContiguous(gamma) ||
    !isContiguous(gradOutput)
  ) {
    throw new Error('rmsnormGradInput: shape or stride mismatch');
  }
  const out = allocateContiguousBuffer(rt, rows * cols, dt);
  dispatchRmsNormGradInputD(
    rt,
    x.buffer,
    gamma.buffer,
    cols,
    gradOutput.buffer,
    out,
    rows,
    cols,
    epsilon,
    dt,
  );
  return makeContiguousTensor([rows, cols], out, dt);
}

/** ∂L/∂γ (`[cols]`) for {@link rmsnorm}. */
export function rmsnormGradGamma(
  rt: TensorRuntime,
  gradOutput: Tensor,
  x: Tensor,
  epsilon = 1e-6,
): Tensor {
  assertSameShape(gradOutput, x);
  assertFloatDtype(gradOutput);
  assertFloatDtype(x);
  assertSameDtype(gradOutput, x);
  assertRank(x, 2);
  const rows = x.shape[0];
  const cols = x.shape[1];
  const dt = floatDtypeOf(x);
  if (!isContiguous(x) || !isContiguous(gradOutput)) {
    throw new Error('rmsnormGradGamma: non-contiguous tensors not supported');
  }
  const out = allocateContiguousBuffer(rt, cols, dt);
  dispatchRmsNormGradGammaD(rt, x.buffer, gradOutput.buffer, out, rows, cols, epsilon, dt);
  return makeContiguousTensor([cols], out, dt);
}

/**
 * Transpose a **rank-2** row-major tensor. Input `[rows, cols]` → output
 * `[cols, rows]`. Throws on rank ≠ 2.
 */
export function transpose(rt: TensorRuntime, a: Tensor): Tensor {
  assertRank(a, 2);
  assertFloatDtype(a);
  const dt = floatDtypeOf(a);
  const rows = a.shape[0];
  const cols = a.shape[1];
  const out = allocateContiguousBuffer(rt, rows * cols, dt);
  dispatchTransposeD(rt, a.buffer, out, rows, cols, dt);
  return makeContiguousTensor([cols, rows], out, dt);
}

/**
 * Embedding gather: lookup **`weights[vocab × embed]`** rows at **`indices[tokens]`** (**`dtype: 'u32'`**).
 */
export function embedding(rt: TensorRuntime, weights: Tensor, indices: Tensor): Tensor {
  assertFloatDtype(weights);
  assertDtype(indices, 'u32');
  assertRank(weights, 2);
  assertRank(indices, 1);
  const dt = floatDtypeOf(weights);
  const embedDim = weights.shape[1];
  const tokens = indices.shape[0];
  const outBuf = allocateContiguousBuffer(rt, tokens * embedDim, dt);
  dispatchEmbeddingGatherD(rt, weights.buffer, indices.buffer, outBuf, tokens, embedDim, dt);
  return makeContiguousTensor([tokens, embedDim], outBuf, dt);
}

/**
 * Atomic scatter-add **`gradTokens[token×embed]`** into a fresh dense **`gradWeights[vocab×embed]`** buffer
 * (**must start zeros** — this helper allocates **`createZeros`** before dispatch).
 */
export function scatterEmbeddingAccum(
  rt: TensorRuntime,
  indices: Tensor,
  gradTokens: Tensor,
  vocabSize: number,
): Tensor {
  assertDtype(indices, 'u32');
  assertFloatDtype(gradTokens);
  assertRank(indices, 1);
  assertRank(gradTokens, 2);
  const tokens = indices.shape[0];
  const embedDim = gradTokens.shape[1];
  const dt = floatDtypeOf(gradTokens);
  if (gradTokens.shape[0] !== tokens) {
    throw new Error(
      `scatterEmbeddingAccum: gradTokens shape[0] (${String(gradTokens.shape[0] ?? '')}) must equal ` +
        `indices length (${String(tokens)})`,
    );
  }
  const accumulator = createZeros(rt, [vocabSize, embedDim], dt);
  dispatchEmbeddingScatterAccumD(
    rt,
    gradTokens.buffer,
    indices.buffer,
    accumulator.buffer,
    tokens,
    embedDim,
    dt,
  );
  return accumulator;
}

/** Per‑row softmax cross‑entropy (**`targets`** length must match **`logits`** rows). */
export function crossEntropyRowsFromLogits(
  rt: TensorRuntime,
  logits: Tensor,
  targets: Tensor,
): Tensor {
  assertFloatDtype(logits);
  assertDtype(targets, 'u32');
  assertRank(logits, 2);
  assertRank(targets, 1);
  const dt = floatDtypeOf(logits);
  const rows = logits.shape[0];
  const cols = logits.shape[1];
  if (targets.shape[0] !== rows) {
    throw new Error(
      `crossEntropyRowsFromLogits: logits rows (${String(rows)}) must match targets (${String(targets.shape[0] ?? '')})`,
    );
  }
  const outBuf = allocateContiguousBuffer(rt, rows, dt);
  dispatchCrossEntropyFromLogitsD(rt, logits.buffer, targets.buffer, outBuf, rows, cols, dt);
  return makeContiguousTensor([rows], outBuf, dt);
}

/**
 * ∂Σ_k **`upstream`**[k]·CE_softmax(...) / ∂logits from row‑wise **`probs`** (softmax of logits).
 */
export function crossEntropyLogitsGrad(
  rt: TensorRuntime,
  probs: Tensor,
  targets: Tensor,
  upstreamPerRow: Tensor,
): Tensor {
  assertFloatDtype(probs);
  assertFloatDtype(upstreamPerRow);
  assertSameDtype(probs, upstreamPerRow);
  assertDtype(targets, 'u32');
  assertRank(probs, 2);
  assertRank(targets, 1);
  assertRank(upstreamPerRow, 1);
  const rows = probs.shape[0];
  const cols = probs.shape[1];
  const dt = floatDtypeOf(probs);
  if (targets.shape[0] !== rows || upstreamPerRow.shape[0] !== rows) {
    throw new Error('crossEntropyLogitsGrad: shapes rows mismatch');
  }
  const outBuf = allocateContiguousBuffer(rt, rows * cols, dt);
  dispatchCrossEntropyLogitsGradD(
    rt,
    probs.buffer,
    targets.buffer,
    upstreamPerRow.buffer,
    outBuf,
    rows,
    cols,
    dt,
  );
  return makeContiguousTensor([rows, cols], outBuf, dt);
}

/** Sum **`[length]`** into a **`[1]`** buffer ( **`dispatchReduceSum`** with one row ). */
export function sumFlat(rt: TensorRuntime, a: Tensor): Tensor {
  assertFloatDtype(a);
  assertRank(a, 1);
  const dt = floatDtypeOf(a);
  const n = a.shape[0];
  const collapsed = reshape(a, [1, n]);
  const outBuf = allocateContiguousBuffer(rt, 1, dt);
  dispatchReduceSumD(rt, collapsed.buffer, outBuf, 1, n, dt);
  return makeContiguousTensor([1], outBuf, dt);
}

/** Host-supplied AdamW scalars including bias corrections `1 - β₁ᵗ` and `1 - β₂ᵗ`. */
export type AdamWStepHyperparams = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  biasCorrection1: number;
  biasCorrection2: number;
};

/**
 * Fused AdamW step: **`param`**, **`moment1`**, **`moment2`** updated in place; **`grad`** read-only.
 * All tensors must be contiguous **`f32`** with identical shape.
 */
export function adamWStepInPlace(
  rt: TensorRuntime,
  param: Tensor,
  grad: Tensor,
  moment1: Tensor,
  moment2: Tensor,
  hyper: AdamWStepHyperparams,
): void {
  assertDtype(param, 'f32');
  assertDtype(grad, 'f32');
  assertDtype(moment1, 'f32');
  assertDtype(moment2, 'f32');
  assertSameShape(param, grad);
  assertSameShape(param, moment1);
  assertSameShape(param, moment2);
  if (
    !isContiguous(param) ||
    !isContiguous(grad) ||
    !isContiguous(moment1) ||
    !isContiguous(moment2)
  ) {
    throw new Error('adamWStepInPlace: non-contiguous tensors not supported');
  }
  dispatchAdamWStep(
    rt.ctx,
    rt.registry,
    param.buffer,
    grad.buffer,
    moment1.buffer,
    moment2.buffer,
    param.length,
    hyper,
  );
}

/**
 * GPU copy of a contiguous **f32** span. **Both** tensors must use storage buffers with
 * **`COPY_SRC` / `COPY_DST`** (the usual **`createZeros`** / **`createFromArray`** allocation).
 */
export function copyF32Elems(
  rt: TensorRuntime,
  src: Tensor,
  srcElemOffset: number,
  dst: Tensor,
  dstElemOffset: number,
  elemCount: number,
): void {
  assertDtype(src, 'f32');
  assertDtype(dst, 'f32');
  if (!Number.isInteger(elemCount) || elemCount < 0) {
    throw new Error(
      `copyF32Elems: elemCount must be a non-negative integer (got ${String(elemCount)})`,
    );
  }
  if (
    !Number.isInteger(srcElemOffset) ||
    !Number.isInteger(dstElemOffset) ||
    srcElemOffset < 0 ||
    dstElemOffset < 0
  ) {
    throw new Error('copyF32Elems: offsets must be non-negative integers');
  }
  if (srcElemOffset + elemCount > src.length || dstElemOffset + elemCount > dst.length) {
    throw new Error(
      `copyF32Elems: range out of bounds (src ${String(srcElemOffset)}+${String(elemCount)} > ${String(
        src.length,
      )}, dst ${String(dstElemOffset)}+${String(elemCount)} > ${String(dst.length)})`,
    );
  }
  if (elemCount === 0) {
    return;
  }
  const encoder = rt.ctx.device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    src.buffer,
    srcElemOffset * 4,
    dst.buffer,
    dstElemOffset * 4,
    elemCount * 4,
  );
  rt.ctx.queue.submit([encoder.finish()]);
}

/**
 * Row-major **view** of the first **`prefixRows`** rows of **`t`** (**`[rows, cols]` → `[prefixRows, cols]`**).
 * Shares **`t.buffer`**; destroying either tensor invalidates both (same contract as {@link reshape}).
 */
export function view2dRowPrefix(t: Tensor, prefixRows: number): Tensor {
  assertRank(t, 2);
  const rows = t.shape[0];
  const cols = t.shape[1];
  if (!Number.isInteger(prefixRows) || prefixRows < 0 || prefixRows > rows) {
    throw new Error(
      `view2dRowPrefix: prefixRows must be in [0, ${String(rows)}] (got ${String(prefixRows)})`,
    );
  }
  if (!isContiguous(t)) {
    throw new Error('view2dRowPrefix: non-contiguous tensor');
  }
  return {
    shape: [prefixRows, cols],
    strides: computeContiguousStrides([prefixRows, cols]),
    dtype: t.dtype,
    device: t.device,
    buffer: t.buffer,
    length: prefixRows * cols,
  };
}

/** Writes **`scalar[0]`** into **`length`** elements — used by **`sumFlat`** autograd broadcasts. */
export function broadcastFillFromScalar(rt: TensorRuntime, scalar: Tensor, length: number): Tensor {
  assertFloatDtype(scalar);
  assertRank(scalar, 1);
  if (scalar.shape[0] !== 1) {
    throw new Error('broadcastFillFromScalar: expects scalar tensor [1]');
  }
  if (!Number.isInteger(length) || length <= 0) {
    throw new Error(`broadcastFillFromScalar: length must be positive (got ${String(length)})`);
  }
  const dt = floatDtypeOf(scalar);
  const outBuf = allocateContiguousBuffer(rt, length, dt);
  dispatchBroadcastFillScalarD(rt, scalar.buffer, outBuf, length, dt);
  return makeContiguousTensor([length], outBuf, dt);
}

/**
 * Pure metadata reshape: the returned tensor **shares** `a.buffer`. No GPU
 * runtime needed — this neither allocates nor dispatches.
 *
 * Throws when the new shape's product differs from `a.length` or when `a` has
 * non-contiguous strides (views beyond reshape are out of scope for
 * `tensor-bootstrap`). Because the buffer is shared, calling
 * {@link import('./tensor').destroy} on either tensor invalidates both.
 */
export function reshape(a: Tensor, newShape: readonly number[]): Tensor {
  const newLength = productOfShape(newShape);
  if (newLength !== a.length) {
    throw new Error(
      `reshape: element count mismatch ` +
        `(current=${String(a.length)} from [${a.shape.join(',')}], ` +
        `requested=${String(newLength)} from [${newShape.join(',')}])`,
    );
  }
  if (!isContiguous(a)) {
    throw new Error('reshape: non-contiguous strides not supported in tensor-bootstrap');
  }
  return {
    shape: newShape,
    strides: computeContiguousStrides(newShape),
    dtype: a.dtype,
    device: a.device,
    buffer: a.buffer,
    length: a.length,
  };
}
