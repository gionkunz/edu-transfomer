/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Float32Array index reads are typed `number | undefined` under
 * `noUncheckedIndexedAccess`, but every index here is bounded by
 * the just-allocated array's known length, so the assertions are safe.
 */

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
        sum += a[i * K + k]! * b[k * N + j]!;
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
    out[i] = a[i]! + b[i]!;
  }
  return out;
}

export function cpuMul(a: Float32Array, b: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = a[i]! * b[i]!;
  }
  return out;
}

export function cpuScale(a: Float32Array, scale: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = a[i]! * scale;
  }
  return out;
}

export function cpuGelu(a: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = geluScalar(a[i]!);
  }
  return out;
}

/** Elementwise `grad_out ⊙ gelu′(inp)` — matches `core.gelu_backward`. */
export function cpuGeluBackward(gradOut: Float32Array, inp: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = gradOut[i]! * geluDerivativeScalar(inp[i]!);
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
      acc += data[base + j]!;
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
      const v = data[base + j]!;
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
      const v = data[base + j]!;
      if (v > m) m = v;
    }
    let s = 0;
    for (let j = 0; j < cols; j++) {
      const e = Math.exp(data[base + j]! - m);
      out[base + j] = e;
      s += e;
    }
    for (let j = 0; j < cols; j++) {
      out[base + j] = out[base + j]! / s;
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
      const v = data[base + j]!;
      sumSq += v * v;
    }
    const inv = 1 / Math.sqrt(sumSq / cols + epsilon);
    for (let j = 0; j < cols; j++) {
      out[base + j] = data[base + j]! * inv * gamma[j]!;
    }
  }
  return out;
}
