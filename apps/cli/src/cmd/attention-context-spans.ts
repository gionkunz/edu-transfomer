export type ContextTokenSpan = {
  readonly text: string;
  /** Absolute token index in the full sequence (prompt + generated). */
  readonly tokenIndex: number;
};

/** Build one display span per token id using the same piece decoder as the inspector. */
export function buildContextTokenSpans(
  promptIds: readonly number[],
  generatedIds: readonly number[],
  displayFor: (id: number) => string,
): ContextTokenSpan[] {
  const spans: ContextTokenSpan[] = [];
  let idx = 0;
  for (const id of promptIds) {
    spans.push({ text: displayFor(id), tokenIndex: idx });
    idx += 1;
  }
  for (const id of generatedIds) {
    spans.push({ text: displayFor(id), tokenIndex: idx });
    idx += 1;
  }
  return spans;
}

/**
 * Keep trailing token spans whose concatenated length fits **`maxChars`**, never splitting a span.
 * Always retains at least the last span when the budget is positive.
 */
export function tailSliceTokenSpans<T extends { readonly text: string }>(
  spans: readonly T[],
  maxChars: number,
): readonly T[] {
  if (maxChars <= 0 || spans.length === 0) return [];
  const out: T[] = [];
  let total = 0;
  for (let i = spans.length - 1; i >= 0; i--) {
    const span = spans[i];
    if (span === undefined) continue;
    const nextTotal = total + span.text.length;
    if (nextTotal > maxChars && out.length > 0) break;
    out.unshift(span);
    total = nextTotal;
    if (total >= maxChars) break;
  }
  return out;
}
