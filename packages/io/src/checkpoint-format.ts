/** Packed checkpoints: `.bin` blob + JSON sidecar (see `docs/project.md` Component 9). */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import {
  bytesPerElement,
  packBf16FromFloat32,
  packFp16FromFloat32,
  unpackBf16ToFloat32,
  unpackFp16ToFloat32,
  type FloatDType,
} from '@llm-transformer-diy/tensor';

import { readJsonFile } from './config-loader';

/** Stored weight dtype in **`model.json`**; omit on legacy manifests (**`f32`** implied). */
export type CheckpointDtype = FloatDType;

export interface TensorManifestEntry {
  readonly offset: number;
  readonly shape: readonly number[];
  readonly dtype: CheckpointDtype;
}

export interface PackedCheckpointSidecar {
  readonly config: unknown;
  readonly tensors: ReadonlyRecord<string, TensorManifestEntry>;
  readonly metadata?: Record<string, unknown>;
}

type ReadonlyRecord<K extends string | number | symbol, V> = Readonly<Record<K, V>>;

export interface TensorPayload {
  readonly shape: readonly number[];
  readonly data: Float32Array;
  /** On-disk dtype; default **`f32`**. Logical values are always **`Float32Array`**. */
  readonly dtype?: CheckpointDtype;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

/** Element count for a dense row-major tensor; `[]` is treated as scalar (1 element). */
export function tensorElementCount(shape: readonly number[]): number {
  if (shape.length === 0) {
    return 1;
  }
  let count = 1;
  for (const dim of shape) {
    assert(Number.isFinite(dim), `checkpoint: invalid shape dimension ${String(dim)}`);
    assert(
      Number.isInteger(dim),
      `checkpoint: shape dimension must be integer, got ${String(dim)}`,
    );
    assert(dim > 0, `checkpoint: shape dimension must be positive, got ${String(dim)}`);
    count *= dim;
    assert(Number.isFinite(count), 'checkpoint: shape product overflow');
  }
  return count;
}

/** Logical tensor byte length on disk (no tail padding). */
export function tensorRawByteLength(shape: readonly number[], dtype: CheckpointDtype): number {
  const n = tensorElementCount(shape);
  return n * bytesPerElement(dtype);
}

/**
 * Each tensor region is followed by zero padding so the next tensor starts at a **4-byte**
 * boundary (required for **`f32`** slices). Pure **`f32`** checkpoints thus stay **byte-identical**
 * to the pre–§8 format (payload lengths are already multiples of 4).
 */
export function tensorPaddedStorageSpan(shape: readonly number[], dtype: CheckpointDtype): number {
  const raw = tensorRawByteLength(shape, dtype);
  return (raw + 3) & ~3;
}

function parseManifestDtype(tensorName: string, raw: unknown): CheckpointDtype {
  const v = raw === undefined || raw === null ? 'f32' : raw;
  assert(typeof v === 'string', `checkpoint json: tensors["${tensorName}"].dtype must be a string`);
  assert(
    v === 'f32' || v === 'bf16' || v === 'fp16',
    `checkpoint json: tensors["${tensorName}"].dtype must be f32, bf16, or fp16 (got ${JSON.stringify(v)})`,
  );
  return v;
}

function sortedTensorNames(names: readonly string[]): string[] {
  return [...names].sort((a, b) => a.localeCompare(b));
}

function encodeTensorPayload(name: string, payload: TensorPayload): Uint8Array {
  const dtype = payload.dtype ?? 'f32';
  const expectLen = tensorElementCount(payload.shape);
  assert(
    payload.data.length === expectLen,
    `checkpoint: tensor "${name}" expected ${String(expectLen)} elements, got ${String(payload.data.length)}`,
  );
  if (dtype === 'f32') {
    return new Uint8Array(payload.data.buffer, payload.data.byteOffset, payload.data.byteLength);
  }
  if (dtype === 'bf16') {
    const packed = packBf16FromFloat32(payload.data);
    return new Uint8Array(packed.buffer, packed.byteOffset, packed.byteLength);
  }
  const packed = packFp16FromFloat32(payload.data);
  return new Uint8Array(packed.buffer, packed.byteOffset, packed.byteLength);
}

/** Build `{ bin, sidecar }` with tensors sorted lexicographically by key. */
export function packPackedCheckpoint(payload: {
  readonly config: unknown;
  readonly tensors: ReadonlyRecord<string, TensorPayload>;
  readonly metadata?: Record<string, unknown>;
}): { readonly bin: Uint8Array; readonly sidecar: PackedCheckpointSidecar } {
  const names = sortedTensorNames(Object.keys(payload.tensors));
  assert(names.length > 0, 'checkpoint: at least one tensor is required');

  const manifest: Record<string, TensorManifestEntry> = {};
  const chunks: Uint8Array[] = [];
  let cursor = 0;

  for (const name of names) {
    const payloadTensor = payload.tensors[name];
    assert(payloadTensor !== undefined, `checkpoint: missing tensor ${name}`);
    const dtype = payloadTensor.dtype ?? 'f32';

    assert(cursor % 4 === 0, 'checkpoint: internal offset alignment invariant broken');

    const encoded = encodeTensorPayload(name, payloadTensor);
    chunks.push(encoded);

    manifest[name] = {
      offset: cursor,
      shape: [...payloadTensor.shape],
      dtype,
    };

    const raw = encoded.byteLength;
    const padded = (raw + 3) & ~3;
    cursor += padded;
    const pad = padded - raw;
    if (pad > 0) {
      chunks.push(new Uint8Array(pad));
    }
  }

  const bin = new Uint8Array(cursor);
  let writeAt = 0;
  for (const chunk of chunks) {
    bin.set(chunk, writeAt);
    writeAt += chunk.byteLength;
  }
  assert(writeAt === bin.byteLength, 'checkpoint: concatenation invariant failed');

  const sidecar: PackedCheckpointSidecar = {
    config: payload.config,
    tensors: manifest,
    ...(payload.metadata !== undefined ? { metadata: payload.metadata } : {}),
  };
  return { bin, sidecar } as const;
}

export function validatePackedCheckpointSidecar(value: unknown): PackedCheckpointSidecar {
  assert(value !== null && typeof value === 'object', 'checkpoint json: root must be an object');
  const root = value as Record<string, unknown>;

  assert('config' in root, 'checkpoint json: missing required field `config`');

  assert('tensors' in root, 'checkpoint json: missing required field `tensors`');
  assert(
    root.tensors !== null && typeof root.tensors === 'object' && !Array.isArray(root.tensors),
    'checkpoint json: `tensors` must be a plain object',
  );

  const tensorsIn = root.tensors as Record<string, unknown>;
  const tensorsOut: Record<string, TensorManifestEntry> = {};
  for (const [tensorName, entryUnknown] of Object.entries(tensorsIn)) {
    assert(
      entryUnknown !== null && typeof entryUnknown === 'object' && !Array.isArray(entryUnknown),
      `checkpoint json: tensors["${tensorName}"] must be an object`,
    );
    const entry = entryUnknown as Record<string, unknown>;

    assert(
      'offset' in entry && typeof entry.offset === 'number',
      `checkpoint json: tensors["${tensorName}"].offset missing`,
    );
    assert(
      Number.isFinite(entry.offset) && Number.isInteger(entry.offset),
      `checkpoint json: tensors["${tensorName}"].offset must be finite integer`,
    );

    assert(
      'shape' in entry && Array.isArray(entry.shape),
      `checkpoint json: tensors["${tensorName}"].shape must be array`,
    );
    const shape: number[] = [];
    for (const dimUnknown of entry.shape) {
      assert(
        typeof dimUnknown === 'number',
        `checkpoint json: tensors["${tensorName}"].shape has non-number`,
      );
      assert(
        Number.isInteger(dimUnknown),
        `checkpoint json: tensors["${tensorName}"].shape dims must be integers`,
      );
      shape.push(dimUnknown);
    }

    const dtype = parseManifestDtype(tensorName, 'dtype' in entry ? entry.dtype : 'f32');
    tensorRawByteLength(shape, dtype);

    tensorsOut[tensorName] = {
      offset: entry.offset,
      shape,
      dtype,
    };
  }

  const manifestCount = Object.keys(tensorsOut).length;
  assert(manifestCount > 0, 'checkpoint json: `tensors` map must declare at least one tensor');

  if ('metadata' in root && root.metadata !== undefined) {
    assert(
      root.metadata !== null && typeof root.metadata === 'object' && !Array.isArray(root.metadata),
      'checkpoint json: `metadata` must be plain object',
    );

    const sidecar: PackedCheckpointSidecar = {
      config: root.config,
      tensors: tensorsOut,
      metadata: { ...(root.metadata as Record<string, unknown>) },
    };
    return sidecar;
  }

  return {
    config: root.config,
    tensors: tensorsOut,
  };
}

function validatePackedSidecarAgainstBinary(
  sidecar: PackedCheckpointSidecar,
  binByteLength: number,
): void {
  const names = sortedTensorNames(Object.keys(sidecar.tensors));
  assert(names.length > 0, 'checkpoint manifest: tensors map is empty');

  let expectedOffset = 0;
  const seenOffsets = new Set<number>();

  for (const name of names) {
    const entry = sidecar.tensors[name];
    assert(entry !== undefined, `checkpoint manifest: tensor "${name}" missing`);
    assert(
      Number.isFinite(entry.offset) && Number.isInteger(entry.offset),
      `checkpoint manifest: offset for "${name}" must be finite integer`,
    );
    assert(entry.offset >= 0, `checkpoint manifest: offset for "${name}" must be non-negative`);
    assert(
      entry.offset % 4 === 0,
      `checkpoint manifest: offset for "${name}" must be 4-byte aligned`,
    );
    assert(
      !seenOffsets.has(entry.offset),
      `checkpoint manifest: duplicate offsets at ${String(entry.offset)}`,
    );
    seenOffsets.add(entry.offset);

    const raw = tensorRawByteLength(entry.shape, entry.dtype);
    const padded = tensorPaddedStorageSpan(entry.shape, entry.dtype);
    assert(
      entry.offset === expectedOffset,
      `checkpoint manifest: gap or overlap detected before "${name}": expected offset ${String(expectedOffset)}, manifest has ${String(entry.offset)}`,
    );
    assert(
      entry.offset + raw <= binByteLength,
      `checkpoint manifest: tensor "${name}" overruns blob (${String(entry.offset + raw)} > ${String(binByteLength)})`,
    );
    assert(
      entry.offset + padded <= binByteLength,
      `checkpoint manifest: tensor "${name}" padded region overruns blob`,
    );
    expectedOffset += padded;
  }

  assert(
    expectedOffset === binByteLength,
    `checkpoint manifest: trailing bytes (${String(binByteLength - expectedOffset)} bytes unaccounted for)`,
  );
}

/** Read tensor slices declared in sidecar into fresh **`Float32Array`** copies (16-bit payloads unpacked). */
export function unpackPackedCheckpoint(
  bin: ArrayBufferLike | Uint8Array,
  sidecar: PackedCheckpointSidecar,
): Record<string, Float32Array> {
  const binBytes = bin instanceof Uint8Array ? bin : new Uint8Array(bin);

  validatePackedSidecarAgainstBinary(sidecar, binBytes.byteLength);

  const out: Record<string, Float32Array> = {};
  const names = sortedTensorNames(Object.keys(sidecar.tensors));

  for (const name of names) {
    const entry = sidecar.tensors[name];
    assert(entry !== undefined, `checkpoint unpack: missing manifest for "${name}"`);
    const raw = tensorRawByteLength(entry.shape, entry.dtype);
    const slice = binBytes.subarray(entry.offset, entry.offset + raw);

    if (entry.dtype === 'f32') {
      assert(
        slice.byteLength % 4 === 0,
        `checkpoint unpack: truncated tensor "${name}" (${String(slice.byteLength)} bytes)`,
      );
      assert(
        slice.byteOffset % 4 === 0,
        `checkpoint unpack: misaligned slice for tensor "${name}"`,
      );
      const count = slice.byteLength / 4;
      const copy = new Float32Array(count);
      copy.set(new Float32Array(slice.buffer, slice.byteOffset, count));
      out[name] = copy;
      continue;
    }

    assert(
      slice.byteLength % 2 === 0,
      `checkpoint unpack: truncated 16-bit tensor "${name}" (${String(slice.byteLength)} bytes)`,
    );
    assert(slice.byteOffset % 2 === 0, `checkpoint unpack: misaligned 16-bit slice for "${name}"`);
    const nHalf = slice.byteLength / 2;
    const u16 = new Uint16Array(slice.buffer, slice.byteOffset, nHalf);
    out[name] = entry.dtype === 'bf16' ? unpackBf16ToFloat32(u16) : unpackFp16ToFloat32(u16);
  }

  return out;
}

export async function writePackedCheckpointFiles(opts: {
  readonly binPath: string;
  readonly jsonPath: string;
  readonly config: unknown;
  readonly tensors: ReadonlyRecord<string, TensorPayload>;
  readonly metadata?: Record<string, unknown>;
}): Promise<PackedCheckpointSidecar> {
  const { bin, sidecar } = packPackedCheckpoint({
    config: opts.config,
    tensors: opts.tensors,
    ...(opts.metadata !== undefined ? { metadata: opts.metadata } : {}),
  });

  await mkdir(path.dirname(opts.binPath), { recursive: true });
  await mkdir(path.dirname(opts.jsonPath), { recursive: true });

  await writeFile(opts.binPath, bin);
  const jsonText = `${JSON.stringify(sidecar, null, 2)}\n`;
  await writeFile(opts.jsonPath, jsonText, 'utf8');
  return sidecar;
}

export async function readPackedCheckpointFiles(
  binPath: string,
  jsonPath: string,
): Promise<{
  readonly sidecar: PackedCheckpointSidecar;
  readonly tensors: Record<string, Float32Array>;
}> {
  const rawSidecar = await readJsonFile(jsonPath);
  const sidecar = validatePackedCheckpointSidecar(rawSidecar);
  const binBuf = await readFile(binPath);
  const tensors = unpackPackedCheckpoint(binBuf, sidecar);
  return { sidecar, tensors } as const;
}
