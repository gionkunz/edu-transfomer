/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Float32Array index reads are typed `number | undefined` under
 * `noUncheckedIndexedAccess`, but every index here is bounded by
 * the just-allocated array's known length, so the assertions are safe.
 */

import { packBf16FromFloat32, unpackBf16ToFloat32 } from './bf16-pack';

/** sqrt(2/π); must match elementwise-gelu.wgsl `GELU_TANH_COEF`. */
export const GELU_TANH_COEF = 0.7978845608028654;

/** Must match elementwise-gelu.wgsl `GELU_CUBE_COEF`. */
export const GELU_CUBE_COEF = 0.044715;

/** Scalar GELU using tanh form — keep identical to WGSL `gelu_f32`. */
export function geluScalar(x: number): number {
  const t = Math.tanh(GELU_TANH_COEF * (x + GELU_CUBE_COEF * x * x * x));
  return 0.5 * x * (1 + t);
}

/** Derivative of {@link geluScalar} — keep identical to WGSL `gelu_grad_f32`. */
export function geluDerivativeScalar(x: number): number {
  const u = GELU_TANH_COEF * (x + GELU_CUBE_COEF * x * x * x);
  const t = Math.tanh(u);
  const dtanh = 1 - t * t;
  const dudx = GELU_TANH_COEF * (1 + 3 * GELU_CUBE_COEF * x * x);
  return 0.5 * (1 + t) + 0.5 * x * dtanh * dudx;
}

/** Row-major M×K × K×N → M×N. */
export function cpuMatMul(
  a: Float32Array,
  b: Float32Array,
  M: number,
  N: number,
  K: number,
): Float32Array {
  const c = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += a[i * K + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
  return c;
}

export function cpuTranspose(src: Float32Array, rows: number, cols: number): Float32Array {
  const dst = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      dst[j * rows + i] = src[i * cols + j]!;
    }
  }
  return dst;
}

export function cpuAdd(a: Float32Array, b: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
  return out;
}

export function cpuMul(a: Float32Array, b: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = a[i] * b[i];
  }
  return out;
}

export function cpuScale(a: Float32Array, scale: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = a[i] * scale;
  }
  return out;
}

export function cpuGelu(a: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = geluScalar(a[i]);
  }
  return out;
}

/** Elementwise `grad_out ⊙ gelu′(inp)` — matches `core.gelu_backward`. */
export function cpuGeluBackward(gradOut: Float32Array, inp: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = gradOut[i] * geluDerivativeScalar(inp[i]);
  }
  return out;
}

/** Row-wise sum of row-major [rows, cols] f32 matrix → [rows] f32 vector. */
export function cpuReduceSum(data: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows);
  for (let i = 0; i < rows; i++) {
    const base = i * cols;
    let acc = 0;
    for (let j = 0; j < cols; j++) {
      acc += data[base + j];
    }
    out[i] = acc;
  }
  return out;
}

/**
 * Row-wise max of row-major [rows, cols] f32 matrix → [rows] f32 vector.
 * Each row's accumulator is initialized to `Number.NEGATIVE_INFINITY`,
 * matching the WGSL kernel's `bitcast<f32>(0xff800000u)` initialization.
 */
export function cpuReduceMax(data: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows);
  for (let i = 0; i < rows; i++) {
    const base = i * cols;
    let acc = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < cols; j++) {
      const v = data[base + j];
      if (v > acc) acc = v;
    }
    out[i] = acc;
  }
  return out;
}

/**
 * Numerically stable row-wise softmax of row-major [rows, cols] f32 matrix.
 * Two-pass per row: subtract row max before exp(), then divide by row sum.
 * Same algorithm the WGSL `softmax.wgsl` kernel implements.
 */
export function cpuSoftmax(data: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const base = i * cols;
    let m = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < cols; j++) {
      const v = data[base + j];
      if (v > m) m = v;
    }
    let s = 0;
    for (let j = 0; j < cols; j++) {
      const e = Math.exp(data[base + j] - m);
      out[base + j] = e;
      s += e;
    }
    for (let j = 0; j < cols; j++) {
      out[base + j] = out[base + j] / s;
    }
  }
  return out;
}

/**
 * Jacobian–vector product for row-wise softmax: gradients w.r.t. logits given
 * upstream grad w.r.t. probabilities `y` (matching `core.softmax_backward`).
 */
export function cpuSoftmaxBackward(
  gradOut: Float32Array,
  y: Float32Array,
  rows: number,
  cols: number,
): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const base = i * cols;
    let dot = 0;
    for (let j = 0; j < cols; j++) {
      dot += y[base + j] * gradOut[base + j];
    }
    for (let j = 0; j < cols; j++) {
      out[base + j] = y[base + j] * (gradOut[base + j] - dot);
    }
  }
  return out;
}

/**
 * Row-wise RMSNorm of row-major [rows, cols] f32 matrix with learned scale
 * `gamma` of length `cols`. For each row i:
 *   meanSq = Σ_j input[i,j]^2 / cols
 *   inv    = 1 / sqrt(meanSq + epsilon)
 *   out[i,j] = input[i,j] * inv * gamma[j]
 * Throws if `gamma.length !== cols`. `epsilon` is required (no default) — the
 * dispatch helper carries the `1e-6` default; this CPU reference is shape-
 * agnostic and accepts any positive `epsilon`.
 */
export function cpuRmsNorm(
  data: Float32Array,
  gamma: Float32Array,
  rows: number,
  cols: number,
  epsilon: number,
): Float32Array {
  if (gamma.length !== cols) {
    throw new Error(
      `cpuRmsNorm: gamma length (${String(gamma.length)}) must equal cols (${String(cols)})`,
    );
  }
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const base = i * cols;
    let sumSq = 0;
    for (let j = 0; j < cols; j++) {
      const v = data[base + j];
      sumSq += v * v;
    }
    const inv = 1 / Math.sqrt(sumSq / cols + epsilon);
    for (let j = 0; j < cols; j++) {
      out[base + j] = data[base + j] * inv * gamma[j];
    }
  }
  return out;
}

export function cpuRmsNormGradInput(
  data: Float32Array,
  gamma: Float32Array,
  gradOut: Float32Array,
  rows: number,
  cols: number,
  epsilon: number,
): Float32Array {
  if (gamma.length !== cols) {
    throw new Error('cpuRmsNormGradInput: gamma length mismatch');
  }
  const gradIn = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const base = i * cols;
    let sumSq = 0;
    for (let j = 0; j < cols; j++) {
      const v = data[base + j];
      sumSq += v * v;
    }
    const inv = 1 / Math.sqrt(sumSq / cols + epsilon);
    let sumTerm = 0;
    for (let j = 0; j < cols; j++) {
      const xv = data[base + j];
      sumTerm += xv * gamma[j] * gradOut[base + j];
    }
    const coeff = (inv * inv * inv) / cols;
    for (let j = 0; j < cols; j++) {
      const xv = data[base + j];
      gradIn[base + j] = inv * gamma[j] * gradOut[base + j] - xv * coeff * sumTerm;
    }
  }
  return gradIn;
}

export function cpuRmsNormGradGamma(
  data: Float32Array,
  gradOut: Float32Array,
  rows: number,
  cols: number,
  epsilon: number,
): Float32Array {
  const gradGamma = new Float32Array(cols);
  for (let j = 0; j < cols; j++) {
    let acc = 0;
    for (let i = 0; i < rows; i++) {
      const base = i * cols;
      let sumSq = 0;
      for (let k = 0; k < cols; k++) {
        const v = data[base + k];
        sumSq += v * v;
      }
      const inv = 1 / Math.sqrt(sumSq / cols + epsilon);
      acc += gradOut[base + j] * data[base + j] * inv;
    }
    gradGamma[j] = acc;
  }
  return gradGamma;
}

/** Gather rows from weight matrix [vocab, embed_dim] flat using u32 indices. */
export function cpuEmbeddingGather(
  weights: Float32Array,
  indices: Uint32Array,
  vocabSize: number,
  embedDim: number,
): Float32Array {
  const n = indices.length;
  const out = new Float32Array(n * embedDim);
  const maxIx = vocabSize * embedDim;
  for (let t = 0; t < n; t++) {
    const vid = indices[t];
    const baseOut = t * embedDim;
    const baseW = vid * embedDim;
    if (baseW + embedDim > maxIx) {
      throw new Error(
        `cpuEmbeddingGather: token ${String(t)} vocab id ${String(vid)} exceeds weight table bounds`,
      );
    }
    for (let d = 0; d < embedDim; d++) {
      out[baseOut + d] = weights[baseW + d]!;
    }
  }
  return out;
}

/** Embedding backward: dense gradWeights [vocab*embed_dim] accumulator from token grads. */
export function cpuEmbeddingScatterAccum(
  gradTokens: Float32Array,
  indices: Uint32Array,
  vocabSize: number,
  embedDim: number,
): Float32Array {
  const gradW = new Float32Array(vocabSize * embedDim);
  const n = indices.length;
  for (let t = 0; t < n; t++) {
    const vid = indices[t];
    const bt = t * embedDim;
    const bw = vid * embedDim;
    for (let d = 0; d < embedDim; d++) {
      const ix = bw + d;
      gradW[ix] = (gradW[ix] ?? 0) + gradTokens[bt + d];
    }
  }
  return gradW;
}

/** Per-row cross-entropy from logits (numerically identical to WGSL helper). */
export function cpuCrossEntropyFromLogits(
  logits: Float32Array,
  targets: Uint32Array,
  rows: number,
  cols: number,
): Float32Array {
  const losses = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const base = row * cols;
    let m = logits[base];
    for (let j = 1; j < cols; j++) {
      m = Math.max(m, logits[base + j]);
    }
    let sumexp = 0;
    for (let j = 0; j < cols; j++) {
      sumexp += Math.exp(logits[base + j] - m);
    }
    const t = targets[row];
    losses[row] = Math.log(sumexp) + m - logits[base + t];
  }
  return losses;
}

/**
 * Gradient of weighted sum Σ_i upstream[i] * CE_row(logits[:,], target[i]) w.r.t. logits,
 * assuming ∂CE/∂logits follows softmax probabilities `probs` (row-wise).
 */
export function cpuCrossEntropyLogitsGradFromProbs(
  probs: Float32Array,
  targets: Uint32Array,
  upstream: Float32Array,
  rows: number,
  cols: number,
): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let row = 0; row < rows; row++) {
    const tgt = targets[row];
    const u = upstream[row];
    const base = row * cols;
    for (let col = 0; col < cols; col++) {
      const p = probs[base + col];
      const delta = col === tgt ? u * (p - 1) : u * p;
      out[base + col] = delta;
    }
  }
  return out;
}

/**
 * Reference fused AdamW (matches `adamw-step.wgsl`): updates `theta`, `m`, and `v` in place.
 */
export function cpuAdamWStep(
  theta: Float32Array,
  grad: Float32Array,
  m: Float32Array,
  v: Float32Array,
  n: number,
  lr: number,
  beta1: number,
  beta2: number,
  eps: number,
  weightDecay: number,
  biasCorrection1: number,
  biasCorrection2: number,
): void {
  for (let i = 0; i < n; i++) {
    const g = grad[i];
    const mNew = beta1 * m[i] + (1 - beta1) * g;
    const vNew = beta2 * v[i] + (1 - beta2) * g * g;
    const mHat = mNew / biasCorrection1;
    const vHat = vNew / biasCorrection2;
    const denom = Math.sqrt(vHat) + eps;
    theta[i] = theta[i] - (lr * mHat) / denom - lr * weightDecay * theta[i];
    m[i] = mNew;
    v[i] = vNew;
  }
}

/** Elementwise **f32 → bf16 → f32** rounding (same as GPU load/store). */
export function roundFloat32ToBf16(src: Float32Array): Float32Array {
  return unpackBf16ToFloat32(packBf16FromFloat32(src));
}

export function cpuMatMulBf16(
  a: Float32Array,
  b: Float32Array,
  M: number,
  N: number,
  K: number,
): Float32Array {
  const ar = roundFloat32ToBf16(a);
  const br = roundFloat32ToBf16(b);
  return roundFloat32ToBf16(cpuMatMul(ar, br, M, N, K));
}

export function cpuTransposeBf16(src: Float32Array, rows: number, cols: number): Float32Array {
  return roundFloat32ToBf16(cpuTranspose(roundFloat32ToBf16(src), rows, cols));
}

export function cpuAddBf16(a: Float32Array, b: Float32Array, n: number): Float32Array {
  const ar = roundFloat32ToBf16(a);
  const br = roundFloat32ToBf16(b);
  return roundFloat32ToBf16(cpuAdd(ar, br, n));
}

export function cpuMulBf16(a: Float32Array, b: Float32Array, n: number): Float32Array {
  const ar = roundFloat32ToBf16(a);
  const br = roundFloat32ToBf16(b);
  return roundFloat32ToBf16(cpuMul(ar, br, n));
}

export function cpuScaleBf16(a: Float32Array, scale: number, n: number): Float32Array {
  const ar = roundFloat32ToBf16(a);
  const sr = roundFloat32ToBf16(new Float32Array([scale]))[0] ?? 0;
  return roundFloat32ToBf16(cpuScale(ar, sr, n));
}

export function cpuGeluBf16(a: Float32Array, n: number): Float32Array {
  return roundFloat32ToBf16(cpuGelu(roundFloat32ToBf16(a), n));
}

export function cpuReduceSumBf16(data: Float32Array, rows: number, cols: number): Float32Array {
  return roundFloat32ToBf16(cpuReduceSum(roundFloat32ToBf16(data), rows, cols));
}

export function cpuReduceMaxBf16(data: Float32Array, rows: number, cols: number): Float32Array {
  return roundFloat32ToBf16(cpuReduceMax(roundFloat32ToBf16(data), rows, cols));
}

export function cpuSoftmaxBf16(data: Float32Array, rows: number, cols: number): Float32Array {
  return roundFloat32ToBf16(cpuSoftmax(roundFloat32ToBf16(data), rows, cols));
}

export function cpuRmsNormBf16(
  data: Float32Array,
  gamma: Float32Array,
  rows: number,
  cols: number,
  epsilon: number,
): Float32Array {
  const dr = roundFloat32ToBf16(data);
  const gr = roundFloat32ToBf16(gamma);
  return roundFloat32ToBf16(cpuRmsNorm(dr, gr, rows, cols, epsilon));
}

export function cpuEmbeddingGatherBf16(
  weights: Float32Array,
  indices: Uint32Array,
  vocabSize: number,
  embedDim: number,
): Float32Array {
  const wr = roundFloat32ToBf16(weights);
  return roundFloat32ToBf16(cpuEmbeddingGather(wr, indices, vocabSize, embedDim));
}

export function cpuCrossEntropyFromLogitsBf16(
  logits: Float32Array,
  targets: Uint32Array,
  rows: number,
  cols: number,
): Float32Array {
  const lr = roundFloat32ToBf16(logits);
  return roundFloat32ToBf16(cpuCrossEntropyFromLogits(lr, targets, rows, cols));
}

export function cpuGeluBackwardBf16(
  gradOut: Float32Array,
  inp: Float32Array,
  n: number,
): Float32Array {
  const gr = roundFloat32ToBf16(gradOut);
  const ir = roundFloat32ToBf16(inp);
  return roundFloat32ToBf16(cpuGeluBackward(gr, ir, n));
}

export function cpuSoftmaxBackwardBf16(
  gradOut: Float32Array,
  y: Float32Array,
  rows: number,
  cols: number,
): Float32Array {
  const gr = roundFloat32ToBf16(gradOut);
  const yr = roundFloat32ToBf16(y);
  return roundFloat32ToBf16(cpuSoftmaxBackward(gr, yr, rows, cols));
}

export function cpuRmsNormGradInputBf16(
  data: Float32Array,
  gamma: Float32Array,
  gradOut: Float32Array,
  rows: number,
  cols: number,
  epsilon: number,
): Float32Array {
  const dr = roundFloat32ToBf16(data);
  const gr = roundFloat32ToBf16(gamma);
  const gor = roundFloat32ToBf16(gradOut);
  return roundFloat32ToBf16(cpuRmsNormGradInput(dr, gr, gor, rows, cols, epsilon));
}

export function cpuRmsNormGradGammaBf16(
  data: Float32Array,
  gradOut: Float32Array,
  rows: number,
  cols: number,
  epsilon: number,
): Float32Array {
  const dr = roundFloat32ToBf16(data);
  const gor = roundFloat32ToBf16(gradOut);
  return roundFloat32ToBf16(cpuRmsNormGradGamma(dr, gor, rows, cols, epsilon));
}

export function cpuEmbeddingScatterAccumBf16(
  gradTokens: Float32Array,
  indices: Uint32Array,
  vocabSize: number,
  embedDim: number,
): Float32Array {
  const tr = roundFloat32ToBf16(gradTokens);
  return roundFloat32ToBf16(cpuEmbeddingScatterAccum(tr, indices, vocabSize, embedDim));
}

export function cpuCrossEntropyLogitsGradFromProbsBf16(
  probs: Float32Array,
  targets: Uint32Array,
  upstream: Float32Array,
  rows: number,
  cols: number,
): Float32Array {
  const pr = roundFloat32ToBf16(probs);
  const ur = roundFloat32ToBf16(upstream);
  return roundFloat32ToBf16(cpuCrossEntropyLogitsGradFromProbs(pr, targets, ur, rows, cols));
}

export function cpuBroadcastFillScalarBf16(scalar: number, n: number): Float32Array {
  const v = roundFloat32ToBf16(new Float32Array([scalar]))[0] ?? 0;
  return roundFloat32ToBf16(new Float32Array(n).fill(v));
}
