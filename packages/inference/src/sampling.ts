/** RNG in **`[0, 1)`** — default **`Math.random`**. */
export type Random01 = () => number;

export type SampleOptions = {
  readonly temperature: number;
  /** `0` disables top-k (use full vocabulary after temperature scaling). */
  readonly topK: number;
};

export const DEFAULT_SAMPLE_OPTIONS: SampleOptions = { temperature: 0.8, topK: 40 };

/**
 * GPT-2 style repetition penalty on a **.mutable** logits row: discourages ids that recently
 * appeared in **`tokenIds`** (positive logits ÷ **`penalty`**, negative logits × **`penalty`**).
 *
 * **`penalty`** &gt; **`1`** enables; **`penalty`** ≤ **`1`** is a no-op.
 */
export function applyRepetitionPenalty(
  logits: Float32Array,
  tokenIds: readonly number[],
  penalty: number,
): void {
  if (penalty <= 1 + 1e-12) {
    return;
  }
  const seen = new Set(tokenIds);
  for (const id of seen) {
    if (!Number.isInteger(id) || id < 0 || id >= logits.length) {
      continue;
    }
    const v = logits[id];
    if (v > 0) {
      logits[id] = v / penalty;
    } else {
      logits[id] = v * penalty;
    }
  }
}

export function greedyArgmax(logits: Float32Array): number {
  if (logits.length === 0) {
    throw new Error('greedyArgmax: empty logits');
  }
  let best = 0;
  let bestV = logits[0];
  for (let i = 1; i < logits.length; i++) {
    const v = logits[i];
    if (v > bestV) {
      bestV = v;
      best = i;
    }
  }
  return best;
}

function softmaxMutable(a: Float32Array): void {
  let mx = -Infinity;
  for (let i = 0; i < a.length; i++) {
    mx = Math.max(mx, a[i]);
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    a[i] = Math.exp(a[i] - mx);
    s += a[i];
  }
  for (let i = 0; i < a.length; i++) {
    a[i] /= s;
  }
}

/**
 * Temperature + optional top-k, then categorical draw.
 *
 * With **`temperature ≤ 0`** (or very small), delegates to **`greedyArgmax`**.
 */
export function sampleLogits(rng: Random01, logits: Float32Array, opts: SampleOptions): number {
  if (logits.length === 0) {
    throw new Error('sampleLogits: empty logits');
  }
  if (opts.temperature <= 1e-8) {
    return greedyArgmax(logits);
  }

  const p = new Float32Array(logits);
  const k = opts.topK;
  if (Number.isInteger(k) && k > 0 && k < p.length) {
    const idx: number[] = Array.from({ length: p.length }, (_, i) => i);
    idx.sort((i, j) => p[j] - p[i]);
    const kth = p[idx[k - 1]];
    for (let i = 0; i < p.length; i++) {
      if (p[i] < kth) {
        p[i] = -Infinity;
      }
    }
  }

  const invTau = 1 / opts.temperature;
  for (let i = 0; i < p.length; i++) {
    p[i] *= invTau;
  }
  softmaxMutable(p);

  let r = rng();
  for (let i = 0; i < p.length; i++) {
    r -= p[i];
    if (r <= 0) {
      return i;
    }
  }
  return p.length - 1;
}
