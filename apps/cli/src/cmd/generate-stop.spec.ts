import { describe, expect, it } from 'vitest';

import { DEFAULT_STOP_TEXT, trailingMarkerTokens } from './generate-stop';

// Toy vocabulary: the marker is spread over three tokens, as with the real BPE.
const pieces = ['The', ' end', '.', '\n', '<|', 'endoftext', '|>', 'Once'];
const decode = (ids: readonly number[]): string => ids.map((i) => pieces[i] ?? '?').join('');

describe('trailingMarkerTokens', () => {
  it('is 0 while the marker is absent or incomplete', () => {
    expect(trailingMarkerTokens([0, 1, 2], decode, DEFAULT_STOP_TEXT)).toBe(0);
    expect(trailingMarkerTokens([0, 1, 2, 3, 4, 5], decode, DEFAULT_STOP_TEXT)).toBe(0);
  });

  it('counts exactly the tokens that spell the marker', () => {
    expect(trailingMarkerTokens([0, 1, 2, 3, 4, 5, 6], decode, DEFAULT_STOP_TEXT)).toBe(3);
  });

  it('is 0 for an empty marker or no tokens', () => {
    expect(trailingMarkerTokens([4, 5, 6], decode, '')).toBe(0);
    expect(trailingMarkerTokens([], decode, DEFAULT_STOP_TEXT)).toBe(0);
  });
});
