import { describe, expect, it } from 'vitest';

import { tailSliceForHighlight } from './generate';

describe('tailSliceForHighlight', () => {
  it('returns the full pair when total length fits in the budget', () => {
    const out = tailSliceForHighlight('hello ', 'world', 100);
    expect(out.prefix).toBe('hello ');
    expect(out.last).toBe('world');
  });

  it('preserves the entire last piece and trims the prefix from the head', () => {
    const out = tailSliceForHighlight('aaaaaa', 'XYZ', 6);
    expect(out.last).toBe('XYZ');
    expect(out.prefix).toBe('aaa');
    expect((out.prefix + out.last).length).toBe(6);
  });

  it('drops the prefix and trims the last piece from the head when last alone exceeds the budget', () => {
    const out = tailSliceForHighlight('garbage', 'longLastPiece', 5);
    expect(out.prefix).toBe('');
    expect(out.last).toBe('Piece');
    expect(out.last.length).toBe(5);
  });

  it('returns empty strings when the budget is non-positive', () => {
    const out = tailSliceForHighlight('foo', 'bar', 0);
    expect(out.prefix).toBe('');
    expect(out.last).toBe('');
  });

  it('always keeps the highlighted last piece visible when there is any budget', () => {
    const prefix = 'p'.repeat(1000);
    const last = 'last';
    const out = tailSliceForHighlight(prefix, last, 5);
    expect(out.last).toBe('last');
    expect(out.prefix).toBe('p');
  });
});
