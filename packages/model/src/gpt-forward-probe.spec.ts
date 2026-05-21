import { describe, expect, it } from 'vitest';

import { layerIndexFromProbeStage, trainingBatchStart } from './gpt-forward-probe';

describe('layerIndexFromProbeStage', () => {
  it('parses block index from coarse and detailed stage names', () => {
    expect(layerIndexFromProbeStage('L7_out')).toBe(7);
    expect(layerIndexFromProbeStage('L7_ffn_gelu')).toBe(7);
    expect(layerIndexFromProbeStage('embed')).toBeNull();
  });
});

describe('trainingBatchStart', () => {
  it('matches llm train step 10287 offset', () => {
    const tokenCount = 2_682_752_671;
    const seqLen = 1024;
    expect(trainingBatchStart(tokenCount, seqLen, 10287)).toBe(102_592_251);
  });

  it('matches llm train step 10239 offset', () => {
    const tokenCount = 2_682_752_671;
    const seqLen = 1024;
    expect(trainingBatchStart(tokenCount, seqLen, 10239)).toBe(102_113_547);
  });

  it('wraps modulo corpus length', () => {
    const tokenCount = 10_000;
    const seqLen = 32;
    const a = trainingBatchStart(tokenCount, seqLen, 0);
    const b = trainingBatchStart(tokenCount, seqLen, 100_000);
    expect(a).toBeGreaterThanOrEqual(0);
    expect(b).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThan(tokenCount - seqLen);
  });
});
