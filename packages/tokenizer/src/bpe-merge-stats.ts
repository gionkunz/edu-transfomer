/**
 * Adjacent pair key for byte-level BPE frequency maps (stable joiner).
 *
 * @internal Reference implementation, not part of the production hot path.
 * The production training path under `packages/tokenizer/src/bpe-train-incremental/`
 * keys pair frequencies by `(L, R)` numerically through `BpePairFreqIndex` to avoid
 * the per-pair string allocation this helper performs. This function exists to back
 * the full-rescan reference trainer used by the equivalence and performance tests
 * (`__tests__/bpe-train-incremental.spec.ts`, `__tests__/bpe-train-perf.spec.ts`).
 */
export function bpePairKey(L: number, R: number): string {
  return `${String(L)}\0${String(R)}`;
}

/**
 * Full-corpus adjacent-pair frequency scan (overlapping observations).
 *
 * @internal Reference implementation, not part of the production hot path.
 * Runs in O(N) per call. The production training loop in
 * `packages/tokenizer/src/bpe-train-incremental/` maintains pair frequencies
 * incrementally instead — this helper exists so the equivalence tests can
 * compare merge-by-merge against the simpler full-rescan baseline, and so the
 * performance benchmark has a baseline to measure speedup against.
 */
export function countAdjacentPairsOverlappingUint32(stream: Uint32Array): Map<string, number> {
  const freq = new Map<string, number>();
  for (let i = 0; i < stream.length - 1; i++) {
    const L = stream[i];
    const R = stream[i + 1];
    if (L === undefined || R === undefined) continue;
    const pk = bpePairKey(L, R);
    freq.set(pk, (freq.get(pk) ?? 0) + 1);
  }
  return freq;
}

/**
 * Best pair for BPE training: max count among pairs with count ≥ 2;
 * tie-break smaller L, then smaller R.
 *
 * @internal Reference implementation, not part of the production hot path.
 * Pairs with the typed-array `pickArgmaxByCountTieLR` in
 * `packages/tokenizer/src/bpe-train-incremental/pair-freq-index.ts` as the
 * full-rescan baseline used by the equivalence and performance tests.
 */
export function pickBestMergePair(freq: Map<string, number>): { L: number; R: number } | null {
  let bestFreq = -1;
  let bestL = Infinity;
  let bestR = Infinity;
  let foundPair = false;

  for (const [k, c] of freq) {
    if (c < 2) continue;
    const ix = k.indexOf('\0');
    const L = Number(k.slice(0, ix));
    const R = Number(k.slice(ix + 1));
    const better =
      !foundPair || c > bestFreq || (c === bestFreq && (L < bestL || (L === bestL && R < bestR)));

    if (better) {
      foundPair = true;
      bestFreq = c;
      bestL = L;
      bestR = R;
    }
  }

  if (!foundPair || bestL === Infinity || bestR === Infinity) {
    return null;
  }
  return { L: bestL, R: bestR };
}
