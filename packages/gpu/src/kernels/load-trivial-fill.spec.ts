import { describe, expect, it } from 'vitest';

import { loadTrivialFillWgsl } from './load-trivial-fill';

describe('loadTrivialFillWgsl', () => {
  it('loads WGSL for the bundled fill kernel', () => {
    const src = loadTrivialFillWgsl();
    expect(src).toContain('@compute');
    expect(src).toContain('3.14159265');
    expect(src).toContain('workgroup_size(256)');
  });
});
