import './webgpu-globals';

import { describe, expect, it } from 'vitest';

import {
  createBf16StorageBuffer,
  createFloat32StorageBuffer,
  float32ByteLength,
  readBf16Buffer,
  readFloat32Buffer,
  writeBf16Buffer,
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

  it('writeBf16Buffer / readBf16Buffer round-trip Uint16 payload', async () => {
    const ctx = await GpuContext.create();
    const n = 5;
    const buf = createBf16StorageBuffer(ctx.device, n);
    const src = new Uint16Array([0x3f80, 0x4000, 0xbf80, 0, 0x7c00]);
    writeBf16Buffer(ctx.queue, buf, src);
    const out = await readBf16Buffer(ctx.device, ctx.queue, buf, n);
    expect(out).toEqual(src);
    buf.destroy();
  });
});
