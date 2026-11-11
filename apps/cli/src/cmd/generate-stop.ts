/** Marker TinyStories-style corpora put between documents; the default **`--stop`** text. */
export const DEFAULT_STOP_TEXT = '<|endoftext|>';

/** How many trailing tokens are decoded when looking for the marker. */
const STOP_SCAN_TOKENS = 24;

/**
 * Number of trailing tokens that spell **`marker`** (plus anything sampled in
 * the same tokens), or **0** when the generated text does not end in it yet.
 *
 * The check runs on **decoded text**, not token ids: the tokenizer has no
 * dedicated end-of-text token, so the marker is ordinary text that BPE splits
 * into several pieces — and can split differently depending on what precedes
 * it. Matching the text is independent of that.
 */
export function trailingMarkerTokens(
  generatedIds: readonly number[],
  decode: (ids: readonly number[]) => string,
  marker: string,
): number {
  if (marker.length === 0 || generatedIds.length === 0) return 0;
  const tail = generatedIds.slice(-STOP_SCAN_TOKENS);
  if (!decode(tail).includes(marker)) return 0;
  for (let k = 1; k <= tail.length; k++) {
    if (decode(tail.slice(-k)).includes(marker)) return k;
  }
  return tail.length;
}
