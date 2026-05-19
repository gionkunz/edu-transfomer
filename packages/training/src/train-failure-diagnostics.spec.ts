import { describe, expect, it } from 'vitest';

import { scanFloat32FiniteStats } from './train-failure-diagnostics';

describe('scanFloat32FiniteStats', () => {
  it('reports finite min/max and maxAbs', () => {
    const data = new Float32Array([-3, 0, 2.5, -2.5]);
    expect(scanFloat32FiniteStats(data)).toEqual({
      nonFinite: 0,
      maxAbs: 3,
      min: -3,
      max: 2.5,
    });
  });

  it('counts NaN and Inf separately from finite stats', () => {
    const data = new Float32Array([1, Number.NaN, Number.POSITIVE_INFINITY, -4]);
    const s = scanFloat32FiniteStats(data);
    expect(s.nonFinite).toBe(2);
    expect(s.maxAbs).toBe(4);
    expect(s.min).toBe(-4);
    expect(s.max).toBe(1);
  });

  it('returns NaN stats when every element is non-finite', () => {
    const data = new Float32Array([Number.NaN, Number.POSITIVE_INFINITY]);
    const s = scanFloat32FiniteStats(data);
    expect(s.nonFinite).toBe(2);
    expect(Number.isNaN(s.maxAbs)).toBe(true);
    expect(Number.isNaN(s.min)).toBe(true);
    expect(Number.isNaN(s.max)).toBe(true);
  });
});
