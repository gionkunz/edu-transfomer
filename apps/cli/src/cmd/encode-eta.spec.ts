import { describe, expect, it } from 'vitest';

import {
  mergeProgressPct,
  pessimisticEtaFromRate,
  pessimisticMergeRemaining,
  pessimisticMergeUpperBound,
} from './encode-eta';

describe('encode-eta', () => {
  it('pessimisticMergeUpperBound', () => {
    expect(pessimisticMergeUpperBound(1)).toBe(1);
    expect(pessimisticMergeUpperBound(10)).toBe(9);
    expect(pessimisticMergeUpperBound(1.9)).toBe(1);
  });

  it('pessimisticMergeRemaining clamps', () => {
    expect(pessimisticMergeRemaining(100, 0)).toBe(99);
    expect(pessimisticMergeRemaining(100, 50)).toBe(49);
    expect(pessimisticMergeRemaining(100, 150)).toBe(0);
  });

  it('mergeProgressPct caps', () => {
    expect(mergeProgressPct(100, 0)).toBe(0);
    expect(mergeProgressPct(100, 49)).toBeCloseTo((100 * 49) / 99, 10);
    expect(mergeProgressPct(100, 99)).toBe(100);
    expect(mergeProgressPct(100, 500)).toBe(100);
  });

  it('pessimisticEtaFromRate', () => {
    expect(pessimisticEtaFromRate(1000, null, 0, 2)).toBe(null);
    expect(pessimisticEtaFromRate(1000, 100, 1, 2)).toBe(null);
    expect(pessimisticEtaFromRate(1000, 100, 2, 2)).toBe(10);
    expect(pessimisticEtaFromRate(0, 100, 2, 2)).toBe(null);
  });
});
