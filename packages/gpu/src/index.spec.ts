import { GPU_LAYER_MARKER } from './index.js';
import { describe, expect, it } from 'vitest';

describe('gpu stub', () => {
  it('exports a stable layer marker', () => {
    expect(GPU_LAYER_MARKER).toBe(Symbol.for('@llm-transformer-diy/gpu'));
  });
});
