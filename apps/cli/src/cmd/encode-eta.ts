/**
 * Heuristics for **train** TUI during **`BPETokenizer.encodeUint32Async`** under the
 * **`word-and-punct-v1`** pre-tokenizer policy: encode folds over independent regions, so the
 * remaining wall-time scales with regions remaining and the average outer-pass count per
 * region.
 */

/**
 * Per-region encode ETA. Cost model:
 *
 *   eta_sec = regions_remaining * mean_passes_per_region / passes_per_second
 *
 * Returns `null` until a per-region pass-rate estimate is available and at least one
 * region has fully completed (so `meanPassesPerRegion` is meaningful).
 */
export function perRegionPessimisticEta(
  regionsRemaining: number,
  meanPassesPerRegion: number | null,
  passesPerSec: number | null,
): number | null {
  if (regionsRemaining <= 0) return 0;
  if (
    meanPassesPerRegion === null ||
    !Number.isFinite(meanPassesPerRegion) ||
    meanPassesPerRegion <= 0
  ) {
    return null;
  }
  if (passesPerSec === null || !Number.isFinite(passesPerSec) || passesPerSec <= 1e-9) {
    return null;
  }
  return (regionsRemaining * meanPassesPerRegion) / passesPerSec;
}
