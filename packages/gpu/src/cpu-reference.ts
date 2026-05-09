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
