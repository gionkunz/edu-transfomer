import { describe, expect, it } from 'vitest';

import { BOOTSTRAP_KERNEL_FILL_PI, GPU_LAYER_MARKER } from './index';

describe('package barrel', () => {
  it('exports a stable GPU layer symbol', () => {
    expect(GPU_LAYER_MARKER).toBe(Symbol.for('@llm-transformer-diy/gpu'));
  });

  it('re-exports bootstrap kernel id', () => {
    expect(BOOTSTRAP_KERNEL_FILL_PI).toBe('bootstrap.fill-pi');
  });
});
