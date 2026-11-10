import { describe, expect, it } from 'vitest';

import { MetricSeries } from './metric-series';

describe('MetricSeries', () => {
  it('keeps every sample while under the bucket limit', () => {
    const s = new MetricSeries(8);
    for (let i = 1; i <= 5; i++) s.push(i, i * 10);
    expect(s.snapshot().map((b) => b.mean)).toEqual([10, 20, 30, 40, 50]);
  });

  it('halves resolution instead of dropping history', () => {
    const s = new MetricSeries(8);
    for (let i = 0; i < 1000; i++) s.push(i, i);
    const snap = s.snapshot();
    expect(snap.length).toBeLessThanOrEqual(8);
    expect(snap[0]?.step).toBe(0);
    expect(snap.reduce((n, b) => n + b.count, 0)).toBe(1000);
    expect(snap[0]?.min).toBe(0);
    expect(snap[snap.length - 1]?.max).toBe(999);
    const first = snap[0];
    expect(first?.mean).toBeCloseTo(((first?.min ?? 0) + (first?.max ?? 0)) / 2, 6);
  });

  it('ignores non-finite samples', () => {
    const s = new MetricSeries(8);
    s.push(1, Number.NaN);
    s.push(2, Number.POSITIVE_INFINITY);
    s.push(3, 4);
    expect(s.snapshot()).toHaveLength(1);
  });
});
