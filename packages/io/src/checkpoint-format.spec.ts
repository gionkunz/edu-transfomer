import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  packPackedCheckpoint,
  readPackedCheckpointFiles,
  unpackPackedCheckpoint,
  validatePackedCheckpointSidecar,
  writePackedCheckpointFiles,
} from './checkpoint-format';

describe('packPackedCheckpoint / unpackPackedCheckpoint', () => {
  it('round-trips multiple tensors with lexical blob order', () => {
    const { bin, sidecar } = packPackedCheckpoint({
      config: { hello: 'world' },
      tensors: {
        z_last: { shape: [2, 2], data: Float32Array.from([9, 8, 7, 6]) },
        aa_first: { shape: [4], data: Float32Array.from([1, 2, 3, 4]) },
      },
      metadata: { step: 1 },
    });

    expect(Object.keys(sidecar.tensors)).toEqual(expect.arrayContaining(['aa_first', 'z_last']));
    const aaFirst = sidecar.tensors['aa_first'];
    const zLast = sidecar.tensors['z_last'];
    expect(aaFirst).toBeDefined();
    expect(zLast).toBeDefined();
    expect(aaFirst.offset).toBe(0);
    expect(zLast.offset).toBe(16);
    expect(bin.byteLength).toBe(32);

    const back = unpackPackedCheckpoint(bin, sidecar);
    const backAaFirst = back['aa_first'];
    const backZLast = back['z_last'];
    expect(backAaFirst).toBeDefined();
    expect(backZLast).toBeDefined();
    expect(Array.from(backAaFirst)).toEqual([1, 2, 3, 4]);
    expect(Array.from(backZLast)).toEqual([9, 8, 7, 6]);
  });

  it('requires at least one tensor when packing', () => {
    expect(() =>
      packPackedCheckpoint({
        config: {},
        tensors: {},
      }),
    ).toThrow(/at least one tensor/);
  });

  it('rejects element-count mismatches', () => {
    expect(() =>
      packPackedCheckpoint({
        config: {},
        tensors: {
          bad: {
            shape: [2],
            data: Float32Array.from([1]),
          },
        },
      }),
    ).toThrow(/bad/);
  });

  it('rejects padded blobs that extend beyond the manifest span', () => {
    const { bin, sidecar } = packPackedCheckpoint({
      config: {},
      tensors: { w: { shape: [2], data: Float32Array.from([1, 2]) } },
    });
    const padded = new Uint8Array(bin.byteLength + 8);
    padded.set(bin);
    expect(() => unpackPackedCheckpoint(padded, sidecar)).toThrow(/trailing bytes/);
  });

  it('rejects gap offsets in manifests', () => {
    expect(() =>
      unpackPackedCheckpoint(
        new Uint8Array(24),
        validatePackedCheckpointSidecar({
          config: {},
          tensors: {
            a: { offset: 0, shape: [2], dtype: 'f32' },
            b: { offset: 16, shape: [2], dtype: 'f32' },
          },
        }),
      ),
    ).toThrow(/gap/);
  });

  it('rejects invalid dtype strings', () => {
    expect(() =>
      validatePackedCheckpointSidecar({
        config: {},
        tensors: { w: { offset: 0, shape: [1], dtype: 'f64' as never } },
      }),
    ).toThrow(/dtype/);
  });

  it('defaults missing per-tensor dtype to f32 (legacy manifests)', () => {
    const sidecar = validatePackedCheckpointSidecar({
      config: {},
      tensors: { w: { offset: 0, shape: [2] } },
    });
    expect(sidecar.tensors.w?.dtype).toBe('f32');
  });

  it('packPackedCheckpoint is deterministic for bf16 (bit-identical bins)', () => {
    const tensors = {
      w: {
        shape: [4] as const,
        data: Float32Array.from([1, 2, -0.5, 0.25]),
        dtype: 'bf16' as const,
      },
    };
    const { bin: a } = packPackedCheckpoint({ config: {}, tensors });
    const { bin: b } = packPackedCheckpoint({ config: {}, tensors });
    expect(Buffer.from(a).equals(Buffer.from(b))).toBe(true);
    expect(a.byteLength % 4).toBe(0);
  });

  it('round-trips bf16 then f32 in one blob with tail padding between regions', () => {
    const { bin, sidecar } = packPackedCheckpoint({
      config: {},
      tensors: {
        h: {
          shape: [3],
          data: Float32Array.from([1, 2, 3]),
          dtype: 'bf16',
        },
        f: { shape: [1], data: Float32Array.from([42]) },
      },
    });
    expect(sidecar.tensors.f?.offset).toBe(0);
    expect(sidecar.tensors.h?.offset).toBe(4);
    const back = unpackPackedCheckpoint(bin, sidecar);
    expect(Array.from(back['h']!)).toEqual([1, 2, 3]);
    expect(Array.from(back['f']!)).toEqual([42]);
  });
});

describe('readPackedCheckpointFiles / writePackedCheckpointFiles', () => {
  it('writes and reads paired files under a temp dir', async () => {
    const dir = await mkdtemp(path.join(tmpdir(), 'io-ckpt-'));
    try {
      const binPath = path.join(dir, 'nested', 'model.bin');
      const jsonPath = path.join(dir, 'nested', 'model.json');

      await writePackedCheckpointFiles({
        binPath,
        jsonPath,
        config: { d_model: 4 },
        tensors: {
          weight: {
            shape: [2],
            data: Float32Array.from([-1, 1]),
          },
        },
      });

      const { sidecar, tensors } = await readPackedCheckpointFiles(binPath, jsonPath);
      expect(sidecar.config).toEqual({ d_model: 4 });
      const weight = tensors['weight'];
      expect(weight).toBeDefined();
      expect(Array.from(weight)).toEqual([-1, 1]);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('parses freshly written JSON with readJson-backed reader', async () => {
    const dir = await mkdtemp(path.join(tmpdir(), 'io-ckpt-'));
    try {
      const binPath = path.join(dir, 'w.bin');
      const jsonPath = path.join(dir, 'w.json');
      await writeFile(jsonPath, 'not-json{', 'utf8');
      const { bin } = packPackedCheckpoint({
        config: {},
        tensors: { w: { shape: [1], data: Float32Array.from([3]) } },
      });
      await writeFile(binPath, bin);
      await expect(readPackedCheckpointFiles(binPath, jsonPath)).rejects.toThrow();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
