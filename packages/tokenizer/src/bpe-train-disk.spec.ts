import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  countAdjacentPairsFromU32LeTokenFile,
  rewriteGreedyU32LeTokenFile,
  writeJoinedCorpusAsTokenIdsToFile,
} from './bpe-train-disk';
import { bpePairKey } from './bpe-merge-stats';

/**
 * Build the expected `u32` LE byte stream for a synthetic corpus: each input
 * byte `b` maps to `byteTokenOffset + b`, and a single newline (10) is inserted
 * between successive files.
 */
function expectedU32LeStream(
  fileBytes: readonly Uint8Array[],
  byteTokenOffset: number,
): Uint8Array {
  let total = 0;
  for (let i = 0; i < fileBytes.length; i++) {
    total += fileBytes[i]!.length;
    if (i > 0) total += 1;
  }
  const out = new Uint8Array(total * 4);
  const view = new DataView(out.buffer);
  let pos = 0;
  for (let f = 0; f < fileBytes.length; f++) {
    if (f > 0) {
      view.setUint32(pos * 4, byteTokenOffset + 10, true);
      pos += 1;
    }
    const buf = fileBytes[f]!;
    for (let i = 0; i < buf.length; i++) {
      view.setUint32(pos * 4, byteTokenOffset + buf[i]!, true);
      pos += 1;
    }
  }
  return out;
}

describe('writeJoinedCorpusAsTokenIdsToFile (post per-byte-await removal)', () => {
  it('produces a u32 LE byte stream identical to the byte-by-byte expected encoding', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-disk-write-spec-'));
    try {
      const fA = path.join(dir, 'a.txt');
      const fB = path.join(dir, 'b.txt');
      const fC = path.join(dir, 'c.txt');
      const aBytes = new TextEncoder().encode('hello world\n');
      const bBytes = new TextEncoder().encode('héllo 世界');
      const cBytes = new Uint8Array([0, 1, 2, 3, 4, 5, 255, 254, 253, 10, 13]);
      await writeFile(fA, aBytes);
      await writeFile(fB, bBytes);
      await writeFile(fC, cBytes);

      const out = path.join(dir, 'tokens.u32');
      const S = 7;
      await writeJoinedCorpusAsTokenIdsToFile({
        filePaths: [fA, fB, fC],
        outPath: out,
        byteTokenOffset: S,
      });

      const got = await readFile(out);
      const want = expectedU32LeStream([aBytes, bBytes, cBytes], S);
      expect(got.length).toBe(want.length);
      expect(Array.from(got)).toEqual(Array.from(want));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('emits cumulative onCorpusUtf8Bytes callbacks (final exact total)', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-disk-write-cb-'));
    try {
      const fA = path.join(dir, 'a.txt');
      const fB = path.join(dir, 'b.txt');
      const aBytes = new Uint8Array(300_000).fill(97);
      const bBytes = new Uint8Array(50_000).fill(98);
      await writeFile(fA, aBytes);
      await writeFile(fB, bBytes);

      const out = path.join(dir, 'tokens.u32');
      const callbacks: bigint[] = [];
      await writeJoinedCorpusAsTokenIdsToFile({
        filePaths: [fA, fB],
        outPath: out,
        byteTokenOffset: 2,
        onCorpusUtf8Bytes: (n) => {
          callbacks.push(n);
        },
      });

      const totalBytes = BigInt(aBytes.length + bBytes.length);
      expect(callbacks.length).toBeGreaterThan(0);
      expect(callbacks[callbacks.length - 1]).toBe(totalBytes);
      // Cumulative — each call >= previous.
      for (let i = 1; i < callbacks.length; i++) {
        expect(callbacks[i]! >= callbacks[i - 1]!).toBe(true);
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe('countAdjacentPairsFromU32LeTokenFile (deprecated parity helper still works)', () => {
  it('counts overlapping pairs in a tiny token file the same way the in-memory helper does', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-disk-count-'));
    try {
      const f = path.join(dir, 'tokens.u32');
      // Write a u32 LE stream of [10, 20, 10, 20, 30].
      const out = new Uint8Array(5 * 4);
      const view = new DataView(out.buffer);
      [10, 20, 10, 20, 30].forEach((v, i) => view.setUint32(i * 4, v, true));
      await writeFile(f, out);
      const freq = await countAdjacentPairsFromU32LeTokenFile(f);
      expect(freq.get(bpePairKey(10, 20))).toBe(2);
      expect(freq.get(bpePairKey(20, 10))).toBe(1);
      expect(freq.get(bpePairKey(20, 30))).toBe(1);
      expect(freq.size).toBe(3);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe('rewriteGreedyU32LeTokenFile (deprecated parity helper still works)', () => {
  it('rewrites (L,R) → newId greedy non-overlapping in u32 LE', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-disk-rewrite-'));
    try {
      const inP = path.join(dir, 'in.u32');
      const outP = path.join(dir, 'out.u32');
      const tokens = [10, 20, 10, 20, 10, 30];
      const buf = new Uint8Array(tokens.length * 4);
      const v = new DataView(buf.buffer);
      tokens.forEach((t, i) => v.setUint32(i * 4, t, true));
      await writeFile(inP, buf);
      await rewriteGreedyU32LeTokenFile({ inPath: inP, outPath: outP, L: 10, R: 20, newId: 999 });

      const got = await readFile(outP);
      const expectedTokens = [999, 999, 10, 30];
      expect(got.length).toBe(expectedTokens.length * 4);
      const gotView = new DataView(got.buffer, got.byteOffset, got.byteLength);
      for (let i = 0; i < expectedTokens.length; i++) {
        expect(gotView.getUint32(i * 4, true)).toBe(expectedTokens[i]);
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
