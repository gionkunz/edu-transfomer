import { describe, expect, it } from 'vitest';

import { explainSampling, sampleLogitsWithProbs } from './sampling';

const LOGITS = new Float32Array([0.1, 5.0, 0.2, 0.3, 4.0, -1.0, 3.5]);

describe('explainSampling', () => {
  it('lists the most likely tokens first with raw probabilities summing to 1 with the rest', () => {
    const b = explainSampling(LOGITS, { temperature: 1, topK: 0 }, 3);
    expect(b.rows.map((r) => r.tokenId)).toEqual([1, 4, 6]);
    const listed = b.rows.reduce((a, r) => a + r.rawProbability, 0);
    expect(listed + b.others.rawProbability).toBeCloseTo(1, 6);
    expect(b.keptCount).toBe(LOGITS.length);
    expect(b.rows.every((r) => r.kept)).toBe(true);
  });

  it('final distribution equals what sampleLogitsWithProbs draws from', () => {
    for (const opts of [
      { temperature: 0.8, topK: 3 },
      { temperature: 1.7, topK: 0 },
      { temperature: 0.3, topK: 100 },
      { temperature: 0, topK: 2 },
    ]) {
      const b = explainSampling(LOGITS, opts, 4);
      const { probabilities } = sampleLogitsWithProbs(() => 0.5, LOGITS, opts);
      expect(Array.from(b.final)).toEqual(Array.from(probabilities));
    }
  });

  it('a lower temperature never lowers the top token and never changes the order', () => {
    let previous = 0;
    for (const temperature of [2, 1.5, 1, 0.7, 0.4, 0.1, 0]) {
      const b = explainSampling(LOGITS, { temperature, topK: 0 }, 3);
      expect(b.rows[0]!.temperedProbability).toBeGreaterThanOrEqual(previous);
      expect(b.rows.map((r) => r.tokenId)).toEqual([1, 4, 6]);
      previous = b.rows[0]!.temperedProbability;
    }
    expect(previous).toBe(1);
  });

  it('top-k marks exactly k tokens as kept and gives the others zero', () => {
    const b = explainSampling(LOGITS, { temperature: 1, topK: 2 }, 5);
    expect(b.keptCount).toBe(2);
    expect(b.rows.filter((r) => r.kept).map((r) => r.tokenId)).toEqual([1, 4]);
    for (const r of b.rows) {
      if (!r.kept) expect(r.finalProbability).toBe(0);
      // Temperature alone removes nothing.
      expect(r.temperedProbability).toBeGreaterThan(0);
    }
    expect(b.others.finalProbability).toBe(0);
  });

  it('greedy keeps only the argmax in the final distribution', () => {
    const b = explainSampling(LOGITS, { temperature: 0, topK: 0 }, 3);
    expect(b.rows[0]!.finalProbability).toBe(1);
    expect(b.rows[1]!.finalProbability).toBe(0);
    // Raw probabilities are unaffected by the sampler settings.
    expect(b.rows[1]!.rawProbability).toBeGreaterThan(0);
  });

  it('rejects empty logits and a non-positive topN', () => {
    expect(() => explainSampling(new Float32Array(0), { temperature: 1, topK: 0 }, 3)).toThrow();
    expect(() => explainSampling(LOGITS, { temperature: 1, topK: 0 }, 0)).toThrow();
  });
});
