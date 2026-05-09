import './webgpu-globals';

import { describe, expect, it } from 'vitest';

import {
  createFloat32StorageBuffer,
  float32ByteLength,
  readFloat32Buffer,
  writeFloat32Buffer,
} from './buffer';
import { GpuContext, probeWebGpu } from './device';

describe('float32ByteLength', () => {
  it('uses four bytes per f32 element', () => {
    expect(float32ByteLength(10)).toBe(40);
    expect(float32ByteLength(0)).toBe(0);
  });
});

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('buffer I/O', () => {
  it('writeFloat32Buffer and readFloat32Buffer round-trip', async () => {
    const ctx = await GpuContext.create();
    const buf = createFloat32StorageBuffer(ctx.device, 4);
    const src = new Float32Array([1, 2, 3, 4]);
    writeFloat32Buffer(ctx.queue, buf, src);

    const out = await readFloat32Buffer(ctx.device, ctx.queue, buf, 16);
    expect(out).toEqual(src);

    buf.destroy();
  });
});
