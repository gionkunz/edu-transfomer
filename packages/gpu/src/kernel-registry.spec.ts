import { describe, expect, it } from 'vitest';

import { KernelRegistry } from './kernel-registry';

describe('KernelRegistry', () => {
  it('rejects duplicate kernel ids', () => {
    const registry = new KernelRegistry();
    registry.register('a', {} as GPUComputePipeline);
    expect(() => {
      registry.register('a', {} as GPUComputePipeline);
    }).toThrow(/duplicate kernel id/);
  });

  it('throws when resolving unknown ids', () => {
    const registry = new KernelRegistry();
    expect(() => registry.get('missing')).toThrow(/unknown kernel id/);
  });
});
