/** Packed f32 checkpoints: `.bin` blob + JSON sidecar (see `docs/project.md` Component 9). */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { readJsonFile } from './config-loader';

export type CheckpointDtype = 'f32';

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

/** Byte span for dense f32 tensor bytes. */
export function tensorPackedByteLength(shape: readonly number[]): number {
  const n = tensorElementCount(shape);
  return n * 4;
}

function sortedTensorNames(names: readonly string[]): string[] {
  return [...names].sort((a, b) => a.localeCompare(b));
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
    const { shape, data } = payloadTensor;

    const expectLen = tensorElementCount(shape);
    assert(
      data.length === expectLen,
      `checkpoint: tensor "${name}" expected ${String(expectLen)} elements, got ${String(data.length)}`,
    );

    const bytesNeeded = tensorPackedByteLength(shape);
    assert(cursor % 4 === 0, 'checkpoint: internal offset alignment invariant broken');

    const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    chunks.push(view);
    manifest[name] = {
      offset: cursor,
      shape: [...shape],
      dtype: 'f32',
    };
    cursor += bytesNeeded;
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
    tensorPackedByteLength(shape);

    assert(
      'dtype' in entry && typeof entry.dtype === 'string',
      `checkpoint json: tensors["${tensorName}"].dtype missing`,
    );
    assert(entry.dtype === 'f32', `checkpoint json: tensors["${tensorName}"].dtype must be "f32"`);

    tensorsOut[tensorName] = {
      offset: entry.offset,
      shape,
      dtype: 'f32',
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

    const byteLen = tensorPackedByteLength(entry.shape);
    assert(
      entry.offset === expectedOffset,
      `checkpoint manifest: gap or overlap detected before "${name}": expected offset ${String(expectedOffset)}, manifest has ${String(entry.offset)}`,
    );
    assert(
      entry.offset + byteLen <= binByteLength,
      `checkpoint manifest: tensor "${name}" overruns blob (${String(entry.offset + byteLen)} > ${String(binByteLength)})`,
    );
    expectedOffset += byteLen;
  }

  assert(
    expectedOffset === binByteLength,
    `checkpoint manifest: trailing bytes (${String(binByteLength - expectedOffset)} bytes unaccounted for)`,
  );
}

/** Read tensor slices declared in sidecar into fresh `Float32Array` copies. */
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
    const byteSpan = tensorPackedByteLength(entry.shape);
    const slice = binBytes.subarray(entry.offset, entry.offset + byteSpan);

    assert(
      slice.byteLength % 4 === 0,
      `checkpoint unpack: truncated tensor "${name}" (${String(slice.byteLength)} bytes)`,
    );
    assert(slice.byteOffset % 4 === 0, `checkpoint unpack: misaligned slice for tensor "${name}"`);

    const count = slice.byteLength / 4;
    const copy = new Float32Array(count);
    copy.set(new Float32Array(slice.buffer, slice.byteOffset, count));
    out[name] = copy;
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
