import {
  dispatchAdd,
  dispatchAdamWStep,
  dispatchBroadcastFillScalar,
  dispatchCrossEntropyFromLogits,
  dispatchCrossEntropyLogitsGrad,
  dispatchEmbeddingGather,
  dispatchEmbeddingScatterAccum,
  dispatchGelu,
  dispatchGeluBackward,
  dispatchMatMul,
  dispatchMul,
  dispatchReduceSum,
  dispatchRmsNorm,
  dispatchRmsNormGradGamma,
  dispatchRmsNormGradInput,
  dispatchScale,
  dispatchSoftmax,
  dispatchSoftmaxBackward,
  dispatchTranspose,
} from '@llm-transformer-diy/gpu';

import {
  assertDtype,
  assertRank,
  assertSameShape,
  computeContiguousStrides,
  isContiguous,
  productOfShape,
} from './shape';
import { allocateContiguousBuffer, createZeros, makeContiguousTensor } from './tensor';
import type { Tensor, TensorRuntime } from './types';

/**
 * Matrix multiply for **rank-2** row-major tensors: `a` is `[M, K]`, `b` is
 * `[K, N]`, output is `[M, N]`. Throws on rank or inner-dim mismatch.
 */
export function matmul(rt: TensorRuntime, a: Tensor, b: Tensor): Tensor {
  assertRank(a, 2);
  assertRank(b, 2);
  const M = a.shape[0];
  const K = a.shape[1];
  const N = b.shape[1];
  if (b.shape[0] !== K) {
    throw new Error(
      `matmul: inner dimensions must match (a=[${a.shape.join(',')}], b=[${b.shape.join(',')}])`,
    );
  }
  const out = allocateContiguousBuffer(rt, M * N);
  dispatchMatMul(rt.ctx, rt.registry, a.buffer, b.buffer, out, M, N, K);
  return makeContiguousTensor([M, N], out);
}

/** Elementwise `out[i] = a[i] + b[i]`. Requires equal shapes (no broadcasting). */
export function add(rt: TensorRuntime, a: Tensor, b: Tensor): Tensor {
  assertSameShape(a, b);
  const out = allocateContiguousBuffer(rt, a.length);
  dispatchAdd(rt.ctx, rt.registry, a.buffer, b.buffer, out, a.length);
  return makeContiguousTensor(a.shape, out);
}

/** Elementwise `out[i] = a[i] * b[i]`. Requires equal shapes (no broadcasting). */
export function mul(rt: TensorRuntime, a: Tensor, b: Tensor): Tensor {
  assertSameShape(a, b);
  const out = allocateContiguousBuffer(rt, a.length);
  dispatchMul(rt.ctx, rt.registry, a.buffer, b.buffer, out, a.length);
  return makeContiguousTensor(a.shape, out);
}

/** Elementwise `out[i] = a[i] * k`. */
export function scale(rt: TensorRuntime, a: Tensor, k: number): Tensor {
  const out = allocateContiguousBuffer(rt, a.length);
  dispatchScale(rt.ctx, rt.registry, a.buffer, out, a.length, k);
  return makeContiguousTensor(a.shape, out);
}

/**
 * Row-wise softmax for **rank-2** row-major `[rows, cols]` (numerically stable;
 * matches `core.softmax`).
 */
export function softmax(rt: TensorRuntime, a: Tensor): Tensor {
  assertRank(a, 2);
  if (!isContiguous(a)) {
    throw new Error('softmax: non-contiguous strides not supported');
  }
  const rows = a.shape[0];
  const cols = a.shape[1];
  const out = allocateContiguousBuffer(rt, rows * cols);
  dispatchSoftmax(rt.ctx, rt.registry, a.buffer, out, rows, cols);
  return makeContiguousTensor([rows, cols], out);
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
  if (!isContiguous(gradOutput) || !isContiguous(softmaxOutput)) {
    throw new Error('softmaxBackward: non-contiguous strides not supported');
  }
  const rows = gradOutput.shape[0];
  const cols = gradOutput.shape[1];
  const out = allocateContiguousBuffer(rt, rows * cols);
  dispatchSoftmaxBackward(
    rt.ctx,
    rt.registry,
    gradOutput.buffer,
    softmaxOutput.buffer,
    out,
    rows,
    cols,
  );
  return makeContiguousTensor([rows, cols], out);
}

/** Elementwise GELU activation (tanh formulation; matches `geluScalar` in `@llm-transformer-diy/gpu`). */
export function gelu(rt: TensorRuntime, a: Tensor): Tensor {
  const out = allocateContiguousBuffer(rt, a.length);
  dispatchGelu(rt.ctx, rt.registry, a.buffer, out, a.length);
  return makeContiguousTensor(a.shape, out);
}

/**
 * GELU backward: `grad_in[i] = grad_out[i] * gelu′(input[i])`. Shapes must
 * match elementwise (no broadcasting).
 */
export function geluBackward(rt: TensorRuntime, gradOutput: Tensor, input: Tensor): Tensor {
  assertSameShape(gradOutput, input);
  const out = allocateContiguousBuffer(rt, input.length);
  dispatchGeluBackward(rt.ctx, rt.registry, gradOutput.buffer, input.buffer, out, input.length);
  return makeContiguousTensor(input.shape, out);
}

/** Row-wise **RMSNorm** on rank-2 **`[rows, cols]`** with **`gamma: [cols]`**. */
export function rmsnorm(rt: TensorRuntime, x: Tensor, gamma: Tensor, epsilon = 1e-6): Tensor {
  assertDtype(x, 'f32');
  assertDtype(gamma, 'f32');
  assertRank(x, 2);
  assertRank(gamma, 1);
  const rows = x.shape[0];
  const cols = x.shape[1];
  if (gamma.shape[0] !== cols) {
    throw new Error(`rmsnorm: gamma length ${String(gamma.shape[0])} != d_model ${String(cols)}`);
  }
  if (!isContiguous(x) || !isContiguous(gamma)) {
    throw new Error('rmsnorm: non-contiguous tensors not supported');
  }
  const out = allocateContiguousBuffer(rt, rows * cols);
  dispatchRmsNorm(rt.ctx, rt.registry, x.buffer, gamma.buffer, cols, out, rows, cols, epsilon);
  return makeContiguousTensor([rows, cols], out);
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
  assertDtype(gamma, 'f32');
  assertRank(x, 2);
  assertRank(gamma, 1);
  const rows = x.shape[0];
  const cols = x.shape[1];
  if (
    gamma.shape[0] !== cols ||
    !isContiguous(x) ||
    !isContiguous(gamma) ||
    !isContiguous(gradOutput)
  ) {
    throw new Error('rmsnormGradInput: shape or stride mismatch');
  }
  const out = allocateContiguousBuffer(rt, rows * cols);
  dispatchRmsNormGradInput(
    rt.ctx,
    rt.registry,
    x.buffer,
    gamma.buffer,
    cols,
    gradOutput.buffer,
    out,
    rows,
    cols,
    epsilon,
  );
  return makeContiguousTensor([rows, cols], out);
}

/** ∂L/∂γ (`[cols]`) for {@link rmsnorm}. */
export function rmsnormGradGamma(
  rt: TensorRuntime,
  gradOutput: Tensor,
  x: Tensor,
  epsilon = 1e-6,
): Tensor {
  assertSameShape(gradOutput, x);
  assertRank(x, 2);
  const rows = x.shape[0];
  const cols = x.shape[1];
  if (!isContiguous(x) || !isContiguous(gradOutput)) {
    throw new Error('rmsnormGradGamma: non-contiguous tensors not supported');
  }
  const out = allocateContiguousBuffer(rt, cols);
  dispatchRmsNormGradGamma(
    rt.ctx,
    rt.registry,
    x.buffer,
    gradOutput.buffer,
    out,
    rows,
    cols,
    epsilon,
  );
  return makeContiguousTensor([cols], out);
}

/**
 * Transpose a **rank-2** row-major tensor. Input `[rows, cols]` → output
 * `[cols, rows]`. Throws on rank ≠ 2.
 */
export function transpose(rt: TensorRuntime, a: Tensor): Tensor {
  assertRank(a, 2);
  const rows = a.shape[0];
  const cols = a.shape[1];
  const out = allocateContiguousBuffer(rt, rows * cols);
  dispatchTranspose(rt.ctx, rt.registry, a.buffer, out, rows, cols);
  return makeContiguousTensor([cols, rows], out);
}

/**
 * Embedding gather: lookup **`weights[vocab × embed]`** rows at **`indices[tokens]`** (**`dtype: 'u32'`**).
 */
export function embedding(rt: TensorRuntime, weights: Tensor, indices: Tensor): Tensor {
  assertDtype(weights, 'f32');
  assertDtype(indices, 'u32');
  assertRank(weights, 2);
  assertRank(indices, 1);
  const embedDim = weights.shape[1];
  const tokens = indices.shape[0];
  const outBuf = allocateContiguousBuffer(rt, tokens * embedDim);
  dispatchEmbeddingGather(
    rt.ctx,
    rt.registry,
    weights.buffer,
    indices.buffer,
    outBuf,
    tokens,
    embedDim,
  );
  return makeContiguousTensor([tokens, embedDim], outBuf);
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
  assertDtype(gradTokens, 'f32');
  assertRank(indices, 1);
  assertRank(gradTokens, 2);
  const tokens = indices.shape[0];
  const embedDim = gradTokens.shape[1];
  if (gradTokens.shape[0] !== tokens) {
    throw new Error(
      `scatterEmbeddingAccum: gradTokens shape[0] (${String(gradTokens.shape[0] ?? '')}) must equal ` +
        `indices length (${String(tokens)})`,
    );
  }
  const accumulator = createZeros(rt, [vocabSize, embedDim]);
  dispatchEmbeddingScatterAccum(
    rt.ctx,
    rt.registry,
    gradTokens.buffer,
    indices.buffer,
    accumulator.buffer,
    tokens,
    embedDim,
  );
  return accumulator;
}

/** Per‑row softmax cross‑entropy (**`targets`** length must match **`logits`** rows). */
export function crossEntropyRowsFromLogits(
  rt: TensorRuntime,
  logits: Tensor,
  targets: Tensor,
): Tensor {
  assertDtype(logits, 'f32');
  assertDtype(targets, 'u32');
  assertRank(logits, 2);
  assertRank(targets, 1);
  const rows = logits.shape[0];
  const cols = logits.shape[1];
  if (targets.shape[0] !== rows) {
    throw new Error(
      `crossEntropyRowsFromLogits: logits rows (${String(rows)}) must match targets (${String(targets.shape[0] ?? '')})`,
    );
  }
  const outBuf = allocateContiguousBuffer(rt, rows);
  dispatchCrossEntropyFromLogits(
    rt.ctx,
    rt.registry,
    logits.buffer,
    targets.buffer,
    outBuf,
    rows,
    cols,
  );
  return makeContiguousTensor([rows], outBuf);
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
  assertDtype(probs, 'f32');
  assertDtype(targets, 'u32');
  assertDtype(upstreamPerRow, 'f32');
  assertRank(probs, 2);
  assertRank(targets, 1);
  assertRank(upstreamPerRow, 1);
  const rows = probs.shape[0];
  const cols = probs.shape[1];
  if (targets.shape[0] !== rows || upstreamPerRow.shape[0] !== rows) {
    throw new Error('crossEntropyLogitsGrad: shapes rows mismatch');
  }
  const outBuf = allocateContiguousBuffer(rt, rows * cols);
  dispatchCrossEntropyLogitsGrad(
    rt.ctx,
    rt.registry,
    probs.buffer,
    targets.buffer,
    upstreamPerRow.buffer,
    outBuf,
    rows,
    cols,
  );
  return makeContiguousTensor([rows, cols], outBuf);
}

/** Sum **`[length]`** into a **`[1]`** buffer ( **`dispatchReduceSum`** with one row ). */
export function sumFlat(rt: TensorRuntime, a: Tensor): Tensor {
  assertDtype(a, 'f32');
  assertRank(a, 1);
  const n = a.shape[0];
  const collapsed = reshape(a, [1, n]);
  const outBuf = allocateContiguousBuffer(rt, 1);
  dispatchReduceSum(rt.ctx, rt.registry, collapsed.buffer, outBuf, 1, n);
  return makeContiguousTensor([1], outBuf);
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
  assertDtype(t, 'f32');
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
  assertDtype(scalar, 'f32');
  assertRank(scalar, 1);
  if (scalar.shape[0] !== 1) {
    throw new Error('broadcastFillFromScalar: expects scalar tensor [1]');
  }
  if (!Number.isInteger(length) || length <= 0) {
    throw new Error(`broadcastFillFromScalar: length must be positive (got ${String(length)})`);
  }
  const outBuf = allocateContiguousBuffer(rt, length);
  dispatchBroadcastFillScalar(rt.ctx, rt.registry, scalar.buffer, outBuf, length);
  return makeContiguousTensor([length], outBuf);
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
  assertDtype(a, 'f32');
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
