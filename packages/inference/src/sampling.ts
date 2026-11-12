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

/** True when **`opts`** asks for the single most likely token instead of a random draw. */
function isGreedy(opts: SampleOptions): boolean {
  return opts.temperature <= 1e-8;
}

/**
 * Which ids survive top-k: every id whose logit is at least the **`k`**-th
 * largest (ties at the cut are all kept). Returns **`null`** when top-k is off.
 */
function topKKeepMask(logits: Float32Array, k: number): Uint8Array | null {
  if (!(Number.isInteger(k) && k > 0 && k < logits.length)) {
    return null;
  }
  const idx: number[] = Array.from({ length: logits.length }, (_, i) => i);
  idx.sort((i, j) => logits[j] - logits[i]);
  const kth = logits[idx[k - 1]];
  const keep = new Uint8Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    keep[i] = logits[i] < kth ? 0 : 1;
  }
  return keep;
}

/**
 * The distribution the sampler draws from: top-k mask, temperature, softmax.
 * Greedy settings give a one-hot vector at the argmax. Single source of truth
 * for **`sampleLogitsWithProbs`** and **`explainSampling`**.
 */
function shapedProbabilities(
  logits: Float32Array,
  opts: SampleOptions,
  keep: Uint8Array | null,
): Float32Array {
  if (isGreedy(opts)) {
    const probabilities = new Float32Array(logits.length);
    probabilities[greedyArgmax(logits)] = 1;
    return probabilities;
  }
  const p = new Float32Array(logits);
  if (keep !== null) {
    for (let i = 0; i < p.length; i++) {
      if (keep[i] === 0) {
        p[i] = -Infinity;
      }
    }
  }
  const invTau = 1 / opts.temperature;
  for (let i = 0; i < p.length; i++) {
    p[i] *= invTau;
  }
  softmaxMutable(p);
  return p;
}

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
  if (isGreedy(opts)) {
    const probabilities = shapedProbabilities(logits, opts, null);
    return { tokenId: greedyArgmax(logits), probabilities };
  }

  const p = shapedProbabilities(logits, opts, topKKeepMask(logits, opts.topK));

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

/** One candidate token followed through the sampler's stages. */
export type SamplingBreakdownRow = {
  readonly tokenId: number;
  /** Plain softmax of the model's scores: what the model itself believes. */
  readonly rawProbability: number;
  /** After temperature only (no top-k). */
  readonly temperedProbability: number;
  /** False when top-k removed this token. */
  readonly kept: boolean;
  /** What the sampler draws from: after temperature and top-k. */
  readonly finalProbability: number;
};

/** The sampler's stages for the most likely tokens; see **`explainSampling`**. */
export type SamplingBreakdown = {
  /** The **`topN`** most likely tokens by raw probability, most likely first. */
  readonly rows: readonly SamplingBreakdownRow[];
  /** Probability mass of all tokens not listed in **`rows`**, per stage. */
  readonly others: {
    readonly rawProbability: number;
    readonly temperedProbability: number;
    readonly finalProbability: number;
  };
  /** Number of tokens top-k keeps (the whole vocabulary when top-k is off). */
  readonly keptCount: number;
  /** Full distribution the sampler draws from; equals **`sampleLogitsWithProbs(...).probabilities`**. */
  readonly final: Float32Array;
};

/**
 * Follow the **`topN`** most likely tokens through the sampler: raw softmax,
 * after temperature, top-k membership, final probability. Pure and
 * deterministic; the final distribution is built by the same code the sampler
 * uses, so what is explained is what is drawn from.
 */
export function explainSampling(
  logits: Float32Array,
  opts: SampleOptions,
  topN: number,
): SamplingBreakdown {
  if (logits.length === 0) {
    throw new Error('explainSampling: empty logits');
  }
  if (!Number.isInteger(topN) || topN <= 0) {
    throw new Error(`explainSampling: topN must be a positive integer (got ${String(topN)})`);
  }
  const raw = new Float32Array(logits);
  softmaxMutable(raw);
  const tempered = shapedProbabilities(logits, opts, null);
  const keep = topKKeepMask(logits, opts.topK);
  const final = shapedProbabilities(logits, opts, keep);

  const order: number[] = Array.from({ length: logits.length }, (_, i) => i);
  order.sort((i, j) => logits[j] - logits[i]);
  const rows: SamplingBreakdownRow[] = [];
  for (const tokenId of order.slice(0, topN)) {
    rows.push({
      tokenId,
      rawProbability: raw[tokenId],
      temperedProbability: tempered[tokenId],
      kept: keep === null || keep[tokenId] === 1,
      finalProbability: final[tokenId],
    });
  }
  // Summed directly rather than as "1 - listed", so an empty remainder is exactly 0.
  const others = { rawProbability: 0, temperedProbability: 0, finalProbability: 0 };
  for (const tokenId of order.slice(topN)) {
    others.rawProbability += raw[tokenId];
    others.temperedProbability += tempered[tokenId];
    others.finalProbability += final[tokenId];
  }
  let keptCount = logits.length;
  if (keep !== null) {
    keptCount = 0;
    for (let i = 0; i < keep.length; i++) {
      keptCount += keep[i];
    }
  }
  return {
    rows,
    others,
    keptCount,
    final,
  };
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
