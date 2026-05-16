export type PackedStorageDType = 'f32' | 'bf16' | 'fp16' | 'u32';

/** Scalar storage size in bytes (GPU buffer stride). */
export function bytesPerElement(dtype: PackedStorageDType): number {
  switch (dtype) {
    case 'f32':
    case 'u32':
      return 4;
    case 'bf16':
    case 'fp16':
      return 2;
    default: {
      const _e: never = dtype;
      return _e;
    }
  }
}

/**
 * Round **f32** to **bf16** (bitcast layout: high 16 bits of **f32**), **RNE** on the
 * dropped mantissa bits (bias `0x8000` on the **u32** bit pattern).
 */
export function packBf16FromFloat32(src: Float32Array, dst?: Uint16Array): Uint16Array {
  const out = dst ?? new Uint16Array(src.length);
  if (out.length < src.length) {
    throw new Error('packBf16FromFloat32: dst shorter than src');
  }
  const srcView = new DataView(src.buffer, src.byteOffset, src.byteLength);
  for (let i = 0; i < src.length; i++) {
    const u = srcView.getUint32(i * 4, true);
    out[i] = (u + 0x8000) >>> 16;
  }
  return out;
}

/** Expand **bf16** half-words to **f32** (`bitcast<f32>(u16 << 16)`). */
export function unpackBf16ToFloat32(src: Uint16Array, dst?: Float32Array): Float32Array {
  const out = dst ?? new Float32Array(src.length);
  if (out.length < src.length) {
    throw new Error('unpackBf16ToFloat32: dst shorter than src');
  }
  const outView = new DataView(out.buffer, out.byteOffset, out.byteLength);
  for (let i = 0; i < src.length; i++) {
    outView.setUint32(i * 4, (src[i]! << 16) >>> 0, true);
  }
  return out;
}

const floatBits = new Float32Array(1);
const intBits = new Int32Array(floatBits.buffer);

/**
 * IEEE-754 **binary16** round-to-nearest-even from a **f32** bit pattern (low-level).
 * **`bits`** is an unsigned **IEEE-754 f32** word (e.g. from **`DataView#getUint32`** LE).
 */
export function float32BitsToFloat16Bits(bits: number): number {
  const x = bits >>> 0;
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mantissa = x & 0x007fffff;

  if (exp === 0xff) {
    if (mantissa === 0) {
      return sign | 0x7c00;
    }
    return sign | 0x7e00;
  }

  let exponent = exp - 127 + 15;

  if (exp === 0) {
    if (mantissa === 0) {
      return sign;
    }
    exponent = 1 - 127 + 15;
    while ((mantissa & 0x00800000) === 0) {
      mantissa <<= 1;
      exponent -= 1;
    }
    mantissa &= 0x007fffff;
  }

  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    mantissa |= 0x00800000;
    const shift = 1 + -exponent;
    const t = mantissa >>> shift;
    const fracMask = (1 << shift) - 1;
    const half = sign | (t >>> 13);
    const roundBit = (t >>> 12) & 1;
    const sticky = (t & 0x0fff) !== 0 || (mantissa & fracMask) !== 0 ? 1 : 0;
    return roundBitsRne(half, roundBit, sticky);
  }

  if (exponent >= 31) {
    return sign | 0x7c00;
  }

  const half = sign | (exponent << 10) | (mantissa >>> 13);
  const roundBit = (mantissa >>> 12) & 1;
  const sticky = (mantissa & 0x0fff) !== 0 ? 1 : 0;
  return roundBitsRne(half, roundBit, sticky);
}

function roundBitsRne(half: number, roundBit: number, sticky: number): number {
  let h = half;
  if (roundBit === 0 && sticky === 0) {
    return h & 0xffff;
  }
  if (roundBit !== 0 && (sticky !== 0 || (h & 1) !== 0)) {
    h += 1;
  }
  return h & 0xffff;
}

/** Round **f32** values to IEEE **binary16** (`Uint16` little-endian words). */
export function packFp16FromFloat32(src: Float32Array, dst?: Uint16Array): Uint16Array {
  const out = dst ?? new Uint16Array(src.length);
  if (out.length < src.length) {
    throw new Error('packFp16FromFloat32: dst shorter than src');
  }
  const srcView = new DataView(src.buffer, src.byteOffset, src.byteLength);
  for (let i = 0; i < src.length; i++) {
    const bits = srcView.getUint32(i * 4, true) | 0;
    out[i] = float32BitsToFloat16Bits(bits);
  }
  return out;
}

/** Decode **binary16** half-words to **f32**. */
export function unpackFp16ToFloat32(src: Uint16Array, dst?: Float32Array): Float32Array {
  const out = dst ?? new Float32Array(src.length);
  if (out.length < src.length) {
    throw new Error('unpackFp16ToFloat32: dst shorter than src');
  }
  for (let i = 0; i < src.length; i++) {
    out[i] = float16BitsToFloat32(src[i]!);
  }
  return out;
}

export function float16BitsToFloat32(h: number): number {
  const s = (h >> 15) & 0x0001;
  let e = (h >> 10) & 0x001f;
  let m = h & 0x03ff;

  if (e === 0) {
    if (m === 0) {
      intBits[0] = s << 31;
      return floatBits[0];
    }
    while ((m & 0x0400) === 0) {
      m <<= 1;
      e -= 1;
    }
    e += 1;
    m &= ~0x0400;
  } else if (e === 31) {
    if (m === 0) {
      intBits[0] = (s << 31) | 0x7f800000;
      return floatBits[0];
    }
    intBits[0] = (s << 31) | 0x7fc00000;
    return floatBits[0];
  }

  e = e + (127 - 15);
  intBits[0] = (s << 31) | (e << 23) | (m << 13);
  return floatBits[0];
}
