/**
 * Heuristics for **train** TUI during **`BPETokenizer.encodeUint32Async`**: each outer
 * best-rank pass removes at least one token slot, so merge iterations are bounded above by
 * **`initialTokenCount - 1`** (worst case).
 */

/** At least **`1`** so downstream division/progress avoids zero. */
export function pessimisticMergeUpperBound(initialTokenCount: number): number {
  if (!Number.isFinite(initialTokenCount) || initialTokenCount < 1) {
    return 1;
  }
  const n = Math.floor(initialTokenCount);
  return Math.max(1, n - 1);
}

export function pessimisticMergeRemaining(
  initialTokenCount: number,
  mergeIteration: number,
): number {
  const upper = pessimisticMergeUpperBound(initialTokenCount);
  return Math.max(0, upper - Math.max(0, Math.floor(mergeIteration)));
}

/**
 * Fraction of the **maximum possible token shrink** (`N−1` slots) already merged away.
 * Outer passes can eliminate many tokens at once, so this moves faster than
 * `mergeIteration / (N−1)` (which sticks near 0% on large corpora). Reaches **100%**
 * only if the sequence shrinks to a single token; real encodes usually finish below that.
 */
export function mergeProgressPct(initialTokenCount: number, currentTokenCount: number): number {
  const upper = pessimisticMergeUpperBound(initialTokenCount);
  if (upper <= 0) return 0;
  const eliminated = Math.max(0, initialTokenCount - currentTokenCount);
  return Math.min(100, (100 * eliminated) / upper);
}

export function pessimisticEtaFromRate(
  remainingMerges: number,
  mergesPerSec: number | null,
  mergeRateSamples: number,
  minSamples: number,
): number | null {
  if (
    mergeRateSamples < minSamples ||
    mergesPerSec === null ||
    !Number.isFinite(mergesPerSec) ||
    mergesPerSec <= 1e-9 ||
    remainingMerges <= 0
  ) {
    return null;
  }
  return remainingMerges / mergesPerSec;
}
