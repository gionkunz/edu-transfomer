import { describe, expect, it } from 'vitest';

import {
  applyRepetitionPenalty,
  greedyArgmax,
  sampleLogits,
  sampleLogitsWithProbs,
} from './sampling';

function rngConstant(x: number): () => number {
  return () => x;
}

function sumF32(a: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
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

  it('sampleLogitsWithProbs returns a normalized distribution of length === vocab', () => {
    const l = new Float32Array([1, 0.5, 0.25, -1, 2]);
    const { tokenId, probabilities } = sampleLogitsWithProbs(rngConstant(0.5), l, {
      temperature: 1,
      topK: 0,
    });
    expect(probabilities.length).toBe(l.length);
    for (let i = 0; i < probabilities.length; i++) {
      expect(probabilities[i]).toBeGreaterThanOrEqual(0);
    }
    expect(sumF32(probabilities)).toBeCloseTo(1, 6);
    expect(tokenId).toBeGreaterThanOrEqual(0);
    expect(tokenId).toBeLessThan(l.length);
  });

  it('sampleLogitsWithProbs zeros probabilities outside the top-k mask', () => {
    const l = new Float32Array([0.1, 5.0, 0.2, 0.3, 4.0]);
    const { probabilities } = sampleLogitsWithProbs(rngConstant(0.5), l, {
      temperature: 1,
      topK: 2,
    });
    // top-2 ids are 1 and 4; the rest must be exactly 0.
    expect(probabilities[0]).toBe(0);
    expect(probabilities[2]).toBe(0);
    expect(probabilities[3]).toBe(0);
    expect(probabilities[1]).toBeGreaterThan(0);
    expect(probabilities[4]).toBeGreaterThan(0);
    expect(sumF32(probabilities)).toBeCloseTo(1, 6);
  });

  it('sampleLogitsWithProbs greedy path returns a one-hot at the argmax', () => {
    const l = new Float32Array([0.1, 0.5, 0.2]);
    const { tokenId, probabilities } = sampleLogitsWithProbs(rngConstant(0.999), l, {
      temperature: 0,
      topK: 0,
    });
    expect(tokenId).toBe(1);
    expect(probabilities[0]).toBe(0);
    expect(probabilities[1]).toBe(1);
    expect(probabilities[2]).toBe(0);
    expect(sumF32(probabilities)).toBe(1);
  });

  it('sampleLogits delegates to sampleLogitsWithProbs (token id parity)', () => {
    const l = new Float32Array(5);
    l[3] = 10;
    const r = rngConstant(0.42);
    const idA = sampleLogits(r, l, { temperature: 1, topK: 1 });
    const { tokenId: idB } = sampleLogitsWithProbs(rngConstant(0.42), l, {
      temperature: 1,
      topK: 1,
    });
    expect(idA).toBe(idB);
    expect(idA).toBe(3);
  });
});
