import { describe, expect, it } from 'vitest';

import { buildContextTokenSpans, tailSliceTokenSpans } from './attention-context-spans';
import {
  blockSelectorRows,
  exploreChromeRows,
  panelRows,
  panelTier,
  rowsForFullPanel,
} from './generate-layout-tiers';

describe('tailSliceTokenSpans', () => {
  it('keeps trailing spans within the char budget without splitting tokens', () => {
    const spans = [
      { text: 'aaa', tokenIndex: 0 },
      { text: 'bbb', tokenIndex: 1 },
      { text: 'cc', tokenIndex: 2 },
    ];
    const out = tailSliceTokenSpans(spans, 5);
    expect(out.map((s) => s.text)).toEqual(['bbb', 'cc']);
  });

  it('always keeps at least the last span when budget is positive', () => {
    const spans = [{ text: 'longpiece', tokenIndex: 0 }];
    const out = tailSliceTokenSpans(spans, 3);
    expect(out).toHaveLength(1);
    expect(out[0]?.text).toBe('longpiece');
  });
});

describe('buildContextTokenSpans', () => {
  it('assigns sequential token indices across prompt and generated ids', () => {
    const displayFor = (id: number): string => `t${String(id)}`;
    const spans = buildContextTokenSpans([1, 2], [3], displayFor);
    expect(spans.map((s) => s.tokenIndex)).toEqual([0, 1, 2]);
    expect(spans.map((s) => s.text)).toEqual(['t1', 't2', 't3']);
  });
});

describe('generate layout tiers', () => {
  // An 8-head model: caption row + 8 head rows.
  const full = 9;

  it('hides the panel on narrow terminals', () => {
    expect(panelTier(40, 30, true, full)).toBe('hidden');
    expect(panelRows('hidden', full)).toBe(0);
  });

  it('uses the full panel when the terminal is tall enough', () => {
    const need = rowsForFullPanel(true, full);
    expect(panelTier(need, 100, true, full)).toBe('full');
    expect(panelTier(need - 1, 100, true, full)).toBe('compressed');
    expect(panelRows('full', full)).toBe(full + 2);
  });

  it('compresses on a 24-row terminal and still leaves a body', () => {
    expect(panelTier(24, 80, true, full)).toBe('compressed');
    expect(24 - exploreChromeRows('compressed', true, full)).toBeGreaterThanOrEqual(3);
  });

  it('hides the panel when even the compressed form leaves no body', () => {
    expect(panelTier(16, 80, true, full)).toBe('hidden');
  });

  it('reserves one row for the block selector', () => {
    expect(blockSelectorRows(true)).toBe(1);
    expect(blockSelectorRows(false)).toBe(0);
  });
});
