import { describe, expect, it } from 'vitest';

import {
  packBf16FromFloat32,
  packFp16FromFloat32,
  unpackBf16ToFloat32,
  unpackFp16ToFloat32,
} from './bf16-fp16-pack';

describe('bf16 pack/unpack (CPU)', () => {
  it('round-trips representable integers exactly', () => {
    const src = new Float32Array([0, 1, 2, 1024, -3]);
    const packed = packBf16FromFloat32(src);
    const out = unpackBf16ToFloat32(packed);
    expect(Array.from(out)).toEqual(Array.from(src));
  });

  /**
   * Halfway case for bf16 RNE: choosing between two representable bf16 numbers — result must pick
   * the one whose low mantissa bit is 0 (ties-to-even).
   */
  it('uses round-to-nearest-even on dropped mantissa bits', () => {
    const f32 = new Float32Array(1);
    const u32 = new DataView(f32.buffer);
    u32.setUint32(0, 0x3f800001, true);
    const half = packBf16FromFloat32(f32)[0];
    const back = unpackBf16ToFloat32(new Uint16Array([half]))[0];
    expect(back).toBe(1);
  });
});

describe('fp16 pack/unpack (CPU)', () => {
  it('saturates huge positive values to +Infinity round-trip', () => {
    const src = new Float32Array([1e30]);
    const p = packFp16FromFloat32(src);
    expect(p[0]).toBe(0x7c00);
    const out = unpackFp16ToFloat32(p);
    expect(out[0]).toBe(Number.POSITIVE_INFINITY);
  });

  it('preserves NaN', () => {
    const src = new Float32Array([Number.NaN]);
    const p = packFp16FromFloat32(src);
    const out = unpackFp16ToFloat32(p);
    expect(Number.isNaN(out[0])).toBe(true);
  });

  it('round-trips a subnormal finite value', () => {
    const half = new Uint16Array([0x0001]);
    const up = unpackFp16ToFloat32(half);
    expect(up[0]).toBeGreaterThan(0);
    expect(Number.isFinite(up[0])).toBe(true);
    const down = packFp16FromFloat32(up);
    expect(down[0]).toBe(0x0001);
  });
});
