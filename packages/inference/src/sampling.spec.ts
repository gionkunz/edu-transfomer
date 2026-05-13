import { describe, expect, it } from 'vitest';

import { applyRepetitionPenalty, greedyArgmax, sampleLogits } from './sampling';

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

  it('applyRepetitionPenalty is a no-op for penalty ≤ 1', () => {
    const l = new Float32Array([1, 2, 3]);
    applyRepetitionPenalty(l, [1], 1);
    expect(Array.from(l)).toEqual([1, 2, 3]);
  });

  it('applyRepetitionPenalty can move argmax off a repeating id', () => {
    const l = new Float32Array(4); // logits for ids 0..3
    l[2] = 5;
    l[1] = 3;
    applyRepetitionPenalty(l, [2], 10);
    expect(greedyArgmax(l)).toBe(1);
  });
});
