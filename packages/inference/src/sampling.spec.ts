import { describe, expect, it } from 'vitest';

import { greedyArgmax, sampleLogits } from './sampling';

function rngConstant(x: number): () => number {
  return () => x;
}

describe('sampling (CPU)', () => {
  it('greedyArgmax picks the largest logit', () => {
    const l = new Float32Array([0.1, 0.5, 0.2]);
    expect(greedyArgmax(l)).toBe(1);
  });

  it('sampleLogits is deterministic when one class has all mass after top-k + softmax', () => {
    const l = new Float32Array(5);
    l[2] = 10;
    const id = sampleLogits(rngConstant(0.05), l, { temperature: 1, topK: 1 });
    expect(id).toBe(2);
  });
});
