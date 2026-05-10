import { describe, expect, it } from 'vitest';

import { getCosineLr, type CosineLrSchedule } from './scheduler';

describe('getCosineLr', () => {
  const base: CosineLrSchedule = {
    lrMax: 0.01,
    lrMin: 0.001,
    warmupSteps: 4,
    totalSteps: 20,
  };

  it('ramps linearly during warmup', () => {
    expect(getCosineLr(0, base)).toBeCloseTo(0.01 * (1 / 4));
    expect(getCosineLr(2, base)).toBeCloseTo(0.01 * (3 / 4));
    expect(getCosineLr(3, base)).toBeCloseTo(0.01);
  });

  it('hits lrMin at end of cosine phase', () => {
    expect(getCosineLr(19, base)).toBeCloseTo(0.001, 5);
  });

  it('uses lrMin when totalSteps <= warmupSteps', () => {
    const s: CosineLrSchedule = { lrMax: 0.1, lrMin: 0.02, warmupSteps: 10, totalSteps: 10 };
    expect(getCosineLr(15, s)).toBeCloseTo(0.02);
  });
});
