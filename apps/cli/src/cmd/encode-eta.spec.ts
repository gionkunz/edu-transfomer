import { describe, expect, it } from 'vitest';

import { perRegionPessimisticEta } from './encode-eta';

describe('encode-eta', () => {
  it('perRegionPessimisticEta covers boundary inputs', () => {
    expect(perRegionPessimisticEta(0, 5, 10)).toBe(0);
    expect(perRegionPessimisticEta(100, null, 10)).toBe(null);
    expect(perRegionPessimisticEta(100, 0, 10)).toBe(null);
    expect(perRegionPessimisticEta(100, 5, null)).toBe(null);
    expect(perRegionPessimisticEta(100, 5, 0)).toBe(null);
    expect(perRegionPessimisticEta(100, 5, 10)).toBe(50);
    expect(perRegionPessimisticEta(10, 3, 6)).toBeCloseTo(5);
  });
});
