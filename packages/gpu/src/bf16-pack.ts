export type PackedStorageDType = 'f32' | 'bf16' | 'u32';

/** Scalar storage size in bytes (GPU buffer stride). */
export function bytesPerElement(dtype: PackedStorageDType): number {
  switch (dtype) {
    case 'f32':
    case 'u32':
      return 4;
    case 'bf16':
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
    outView.setUint32(i * 4, (src[i] << 16) >>> 0, true);
  }
  return out;
}
