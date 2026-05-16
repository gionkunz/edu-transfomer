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

/** Pair returned by **`sampleLogitsWithProbs`**: chosen token id and the post-shaped probability vector. */
export type SampleWithProbsResult = {
  readonly tokenId: number;
  /**
   * Distribution actually used for the categorical draw (length === **`logits.length`**):
   * post-temperature, post-top-k softmax. Greedy returns a one-hot vector at the argmax.
   * Ids masked out by top-k SHALL be exactly **`0`**.
   */
  readonly probabilities: Float32Array;
};

/**
 * Temperature + optional top-k softmax shaping followed by a categorical draw.
 *
 * Returns the sampled **`tokenId`** alongside the **`probabilities`** vector that drove the draw,
 * so hosts can inspect "what the sampler saw" without re-implementing softmax/top-k.
 *
 * With **`temperature ≤ 1e-8`** the result is greedy: **`tokenId`** is **`greedyArgmax(logits)`**
 * and **`probabilities`** is a one-hot at that index.
 */
export function sampleLogitsWithProbs(
  rng: Random01,
  logits: Float32Array,
  opts: SampleOptions,
): SampleWithProbsResult {
  if (logits.length === 0) {
    throw new Error('sampleLogitsWithProbs: empty logits');
  }
  if (opts.temperature <= 1e-8) {
    const tokenId = greedyArgmax(logits);
    const probabilities = new Float32Array(logits.length);
    probabilities[tokenId] = 1;
    return { tokenId, probabilities };
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
  let tokenId = p.length - 1;
  for (let i = 0; i < p.length; i++) {
    r -= p[i];
    if (r <= 0) {
      tokenId = i;
      break;
    }
  }
  return { tokenId, probabilities: p };
}

/**
 * Temperature + optional top-k, then categorical draw.
 *
 * With **`temperature ≤ 0`** (or very small), delegates to **`greedyArgmax`**.
 *
 * Thin wrapper over **`sampleLogitsWithProbs`** that drops the probability vector — kept for
 * callers that only need the token id.
 */
export function sampleLogits(rng: Random01, logits: Float32Array, opts: SampleOptions): number {
  return sampleLogitsWithProbs(rng, logits, opts).tokenId;
}
