/**
 * Adjacent pair key for byte-level BPE frequency maps (stable joiner).
 *
 * @deprecated Allocates a fresh string per call, which dominates the BPE
 * training hot path on large corpora (~30 ns per pair × 300 MB × M merges).
 * The production training path under
 * `packages/tokenizer/src/bpe-train-incremental/` keys pair frequencies by
 * `(L, R)` numerically through `BpePairFreqIndex` — use that instead. This
 * helper is retained for fixtures, debug tooling, and parity tests only.
 */
export function bpePairKey(L: number, R: number): string {
  return `${String(L)}\0${String(R)}`;
}

/**
 * @deprecated O(N) per call. The training loop used to invoke this once per
 * merge, producing the `O(N · M)` blow-up that the incremental pipeline in
 * `packages/tokenizer/src/bpe-train-incremental/` fixes. Retained for parity
 * tests against the new path; do not use in new hot-path code.
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
 * @deprecated Linear scan over a `Map<string, number>` whose keys are produced
 * by {@link bpePairKey}; both per-pair string allocation and the full-frequency-map
 * input are dominated by the typed-array `pickArgmaxByCountTieLR` in
 * `packages/tokenizer/src/bpe-train-incremental/pair-freq-index.ts`. Retained
 * for parity tests only.
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
