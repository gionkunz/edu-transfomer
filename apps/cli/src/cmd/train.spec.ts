import { describe, expect, it } from 'vitest';

import { resolveWarmupSteps, TRAIN_DEFAULT_WARMUP_FRACTION } from './train';

describe('resolveWarmupSteps', () => {
  it('defaults to 10% of total steps for 50k run', () => {
    expect(resolveWarmupSteps(50_000, TRAIN_DEFAULT_WARMUP_FRACTION)).toBe(5000);
  });

  it('clamps to at least one warmup step on tiny runs', () => {
    expect(resolveWarmupSteps(5, 0.1)).toBe(1);
  });

  it('never consumes the entire schedule', () => {
    expect(resolveWarmupSteps(100, 0.99)).toBe(99);
  });

  it('honours custom fractions', () => {
    expect(resolveWarmupSteps(10_000, 0.05)).toBe(500);
    expect(resolveWarmupSteps(10_000, 0.2)).toBe(2000);
  });

  it('rejects invalid fractions', () => {
    expect(() => resolveWarmupSteps(100, 0)).toThrow(/0 < warmup < 1/);
    expect(() => resolveWarmupSteps(100, 1)).toThrow(/0 < warmup < 1/);
    expect(() => resolveWarmupSteps(100, -0.1)).toThrow(/0 < warmup < 1/);
  });
});
