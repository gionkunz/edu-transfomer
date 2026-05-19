import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

import {
  type CorpusTokenCacheFingerprint,
  corpusTokenCacheArtifactPaths,
  corpusTokenFingerprintsEqual,
  describeCorpusTokenFingerprintMismatch,
  readCorpusTokenCache,
  writeCachedCorpusTokens,
} from './corpus-token-cache';

describe('corpusTokenFingerprintsEqual', () => {
  function fp(partial: Partial<CorpusTokenCacheFingerprint>): CorpusTokenCacheFingerprint {
    return {
      formatVersion: 1,
      joinBetweenFiles: '\n',
      tokenizerResolved: '/abs/tok.json',
      tokenizerMtimeMs: 1,
      corpusFiles: [],
      ...partial,
    };
  }

  it('matches identical fingerprints', () => {
    const a = fp({
      corpusFiles: [{ path: '/abs/a.txt', size: 3, mtimeMs: 2 }],
    });
    expect(corpusTokenFingerprintsEqual(a, { ...a, corpusFiles: [...a.corpusFiles] })).toBe(true);
  });

  it('rejects mtime drift', () => {
    const a = fp({});
    expect(
      corpusTokenFingerprintsEqual(a, {
        ...a,
        tokenizerMtimeMs: 99,
      }),
    ).toBe(false);
  });

  it('rejects corpus file length mismatch', () => {
    expect(
      corpusTokenFingerprintsEqual(
        fp({ corpusFiles: [{ path: '/a', size: 1, mtimeMs: 1 }] }),
        fp({ corpusFiles: [] }),
      ),
    ).toBe(false);
  });
});

describe('describeCorpusTokenFingerprintMismatch', () => {
  function fp(partial: Partial<CorpusTokenCacheFingerprint>): CorpusTokenCacheFingerprint {
    return {
      formatVersion: 1,
      joinBetweenFiles: '\n',
      tokenizerResolved: '/abs/tok.json',
      tokenizerMtimeMs: 100,
      corpusFiles: [{ path: '/abs/corpus/a.txt', size: 10, mtimeMs: 200 }],
      ...partial,
    };
  }

  it('returns empty array when fingerprints match', () => {
    const a = fp({});
    expect(describeCorpusTokenFingerprintMismatch(a, { ...a })).toEqual([]);
  });

  it('reports tokenizer mtime drift', () => {
    const a = fp({});
    const reasons = describeCorpusTokenFingerprintMismatch(a, { ...a, tokenizerMtimeMs: 101 });
    expect(reasons.some((r) => r.includes('tokenizer mtime'))).toBe(true);
  });

  it('reports corpus path drift with cwd hint', () => {
    const a = fp({});
    const reasons = describeCorpusTokenFingerprintMismatch(a, {
      ...a,
      corpusFiles: [{ path: '/other/corpus/a.txt', size: 10, mtimeMs: 200 }],
    });
    expect(reasons.some((r) => r.includes('path') && r.includes('cwd'))).toBe(true);
  });

  it('caps per-file diff lines', () => {
    const expected = fp({
      corpusFiles: [
        { path: '/abs/0.txt', size: 1, mtimeMs: 1 },
        { path: '/abs/1.txt', size: 1, mtimeMs: 1 },
        { path: '/abs/2.txt', size: 1, mtimeMs: 1 },
        { path: '/abs/3.txt', size: 1, mtimeMs: 1 },
      ],
    });
    const disk = fp({
      corpusFiles: [
        { path: '/abs/0.txt', size: 9, mtimeMs: 1 },
        { path: '/abs/1.txt', size: 9, mtimeMs: 1 },
        { path: '/abs/2.txt', size: 9, mtimeMs: 1 },
        { path: '/abs/3.txt', size: 9, mtimeMs: 1 },
      ],
    });
    const reasons = describeCorpusTokenFingerprintMismatch(expected, disk, {
      maxCorpusFileDiffs: 2,
    });
    expect(reasons.length).toBe(3);
    expect(reasons[2]).toMatch(/and 2 more corpus file mismatch/);
  });
});

describe('readCorpusTokenCache', () => {
  const tempDirs: string[] = [];

  afterEach(async () => {
    await Promise.all(tempDirs.splice(0).map((d) => rm(d, { recursive: true, force: true })));
  });

  async function tempDir(): Promise<string> {
    const d = await mkdtemp(path.join(os.tmpdir(), 'llm-cache-diag-'));
    tempDirs.push(d);
    return d;
  }

  function fp(partial: Partial<CorpusTokenCacheFingerprint>): CorpusTokenCacheFingerprint {
    return {
      formatVersion: 1,
      joinBetweenFiles: '\n',
      tokenizerResolved: '/abs/tok.json',
      tokenizerMtimeMs: 1,
      corpusFiles: [],
      ...partial,
    };
  }

  it('returns miss when meta.json is missing', async () => {
    const dir = await tempDir();
    const base = path.join(dir, 'cache', 'run');
    const result = await readCorpusTokenCache(base, fp({}));
    expect(result.status).toBe('miss');
    if (result.status === 'miss') {
      expect(result.reasons[0]).toMatch(/meta\.json not found/);
    }
  });

  it('returns hit when manifest and ids match fingerprint', async () => {
    const dir = await tempDir();
    const base = path.join(dir, 'cache', 'run');
    const fingerprint = fp({});
    const ids = new Uint32Array([1, 2, 3]);
    await writeCachedCorpusTokens(base, fingerprint, ids, 42);

    const result = await readCorpusTokenCache(base, fingerprint);
    expect(result.status).toBe('hit');
    if (result.status === 'hit') {
      expect(Array.from(result.ids)).toEqual([1, 2, 3]);
      expect(result.corpusUtf16Units).toBe(42);
    }
  });

  it('returns miss when ids payload size is wrong', async () => {
    const dir = await tempDir();
    const base = path.join(dir, 'cache', 'run');
    const fingerprint = fp({});
    await writeCachedCorpusTokens(base, fingerprint, new Uint32Array([1, 2]), 10);
    const paths = corpusTokenCacheArtifactPaths(base);
    await writeFile(paths.idsBin, Buffer.from([0, 1, 2]));

    const result = await readCorpusTokenCache(base, fingerprint);
    expect(result.status).toBe('miss');
    if (result.status === 'miss') {
      expect(result.reasons[0]).toMatch(/size 3 bytes/);
    }
  });

  it('returns miss with fingerprint reasons when tokenizer mtime drifts', async () => {
    const dir = await tempDir();
    const base = path.join(dir, 'cache', 'run');
    const onDisk = fp({ tokenizerMtimeMs: 1 });
    await writeCachedCorpusTokens(base, onDisk, new Uint32Array([7]), 1);

    const result = await readCorpusTokenCache(base, fp({ tokenizerMtimeMs: 2 }));
    expect(result.status).toBe('miss');
    if (result.status === 'miss') {
      expect(result.reasons.some((r) => r.includes('tokenizer mtime'))).toBe(true);
    }
  });

  it('round-trips ids larger than one I/O chunk', async () => {
    const dir = await tempDir();
    const base = path.join(dir, 'cache', 'run');
    const fingerprint = fp({});
    const tokenCount = 64 * 1024 + 17;
    const ids = new Uint32Array(tokenCount);
    for (let i = 0; i < tokenCount; i++) {
      ids[i] = i * 3 + 1;
    }
    await writeCachedCorpusTokens(base, fingerprint, ids, 100);

    const result = await readCorpusTokenCache(base, fingerprint);
    expect(result.status).toBe('hit');
    if (result.status === 'hit') {
      expect(result.ids.length).toBe(tokenCount);
      expect(result.ids[0]).toBe(1);
      expect(result.ids[17]).toBe(17 * 3 + 1);
      expect(result.ids[tokenCount - 1]).toBe((tokenCount - 1) * 3 + 1);
    }
  });

  it('writes manifest readable by readCorpusTokenCache', async () => {
    const dir = await tempDir();
    const base = path.join(dir, 'cache', 'run');
    const fingerprint = fp({
      corpusFiles: [{ path: '/data/a.txt', size: 5, mtimeMs: 9 }],
    });
    await writeCachedCorpusTokens(base, fingerprint, new Uint32Array([99]), 5);
    const paths = corpusTokenCacheArtifactPaths(base);
    const raw = await readFile(paths.metaJson, 'utf8');
    expect(raw).toContain('"tokenCount": 1');
  });
});
