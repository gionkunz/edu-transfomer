import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { BPETokenizer, type BpeEncodeProgressEvent } from './bpe-tokenizer';

function baseVocabMap(specials: readonly string[]): Record<string, number[]> {
  const enc = new TextEncoder();
  const vocab: Record<string, number[]> = {};
  let id = 0;
  for (const s of specials) {
    vocab[String(id)] = Array.from(enc.encode(s));
    id += 1;
  }
  const S = specials.length;
  for (let b = 0; b < 256; b++) {
    vocab[String(S + b)] = [b];
  }
  return vocab;
}

/** Shallow-copy vocab rows, dropping one id — for malformed-load fixtures. */
function vocabWithoutRow(
  base: Readonly<Record<string, readonly number[]>>,
  dropId: number,
): Record<string, number[]> {
  const dropKey = String(dropId);
  const out: Record<string, number[]> = {};
  for (const key of Object.keys(base)) {
    if (key === dropKey) continue;
    const row = base[key];
    if (!row) continue;
    out[key] = [...row];
  }
  return out;
}

describe('BPETokenizer', () => {
  const minVocab = 2 + 256;

  const DEFAULT_SPECIALS = ['<|pad|>', '<|endoftext|>'] as const;

  /**
   * Base vocab (specials + 256 byte tokens) plus one merge id at S+256 with arbitrary bytes
   * (for strict-UTF-8 tests the bytes may be ill-formed).
   */
  function oneMergeVocabWithBytesAtMergeId(spec: {
    specials: readonly string[];
    mergeLeft: number;
    mergeRight: number;
    mergedIdBytes: readonly number[];
  }): { vocab: Record<string, number[]>; merges: [number, number][] } {
    const vocab = baseVocabMap(spec.specials);
    const S = spec.specials.length;
    vocab[String(S + 256)] = [...spec.mergedIdBytes];
    return { vocab, merges: [[spec.mergeLeft, spec.mergeRight]] };
  }

  async function tmpJson(
    contents: Record<string, unknown>,
  ): Promise<{ path: string; cleanup: () => Promise<void> }> {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-test-fixture-'));
    const filePath = path.join(dir, 'tok.json');
    await writeFile(filePath, `${JSON.stringify(contents)}\n`);
    const cleanup = async (): Promise<void> => {
      await rm(dir, { recursive: true, force: true });
    };
    return { path: filePath, cleanup };
  }

  it('throws when save runs before train or load', async () => {
    const { path: filePath, cleanup } = await tmpJson({});
    try {
      await expect(new BPETokenizer().save(filePath)).rejects.toThrow(/train\(\)|load()/);
    } finally {
      await cleanup();
    }
  });

  it('throws when encode runs before train or load', () => {
    const tok = new BPETokenizer();
    expect(() => tok.encode('a')).toThrow(/train\(\)|load()/);
    expect(() => tok.decode([])).toThrow(/train\(\)|load()/);
  });

  it('rejects vocabSize smaller than specials + byte base', () => {
    const tok = new BPETokenizer();
    expect(() => {
      tok.train('anything', minVocab - 1);
    }).toThrow(/less than required minimum/);

    const t2 = new BPETokenizer({ specialTokens: ['<|s|>'] });
    expect(() => {
      t2.train('anything', minVocab - 2);
    }).toThrow(/less than required minimum/);
  });

  it('rejects invalid specialTokens at construction', () => {
    expect(() => new BPETokenizer({ specialTokens: [] })).toThrow(/non-empty/);
    expect(() => new BPETokenizer({ specialTokens: [''] })).toThrow(/empty strings/);
    expect(() => new BPETokenizer({ specialTokens: ['x', 'x'] })).toThrow(
      /duplicate special token/,
    );
  });

  it('round-trips mixed Unicode corpus when trained on matching text', () => {
    const corpus = 'héllo 世界\nping 123';
    const tok = new BPETokenizer();
    tok.train(corpus, 400);
    expect(tok.decode(tok.encode(corpus))).toBe(corpus);
  });

  it('trainFromFiles matches train() on corpus joined like CLI (newline between files)', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-trainfromfiles-parity-'));
    try {
      const f1 = path.join(dir, 'a.txt');
      const f2 = path.join(dir, 'b.txt');
      await writeFile(f1, 'hello ');
      await writeFile(f2, 'world');
      const abs = [f1, f2].map((p) => path.resolve(p));

      const v = 400;
      const tMem = new BPETokenizer();
      const tDisk = new BPETokenizer();
      tMem.train('hello \nworld', v);
      await tDisk.trainFromFiles(abs, v);

      expect(tDisk.vocabSize).toBe(tMem.vocabSize);

      const probes = ['hello \nworld', 'hello ', 'world', 'héllo', ''];
      for (const p of probes) {
        expect(tDisk.encode(p)).toEqual(tMem.encode(p));
      }
      for (const p of probes.filter((x) => x !== '')) {
        expect(tDisk.decode(tDisk.encode(p))).toBe(tMem.decode(tMem.encode(p)));
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('trainAsync agrees with sync train()', async () => {
    const corpus = 'abababcccdd extra';
    const cap = 320;
    const tSync = new BPETokenizer();
    const tAsync = new BPETokenizer();
    tSync.train(corpus, cap);
    await tAsync.trainAsync(corpus, cap);
    expect(tAsync.vocabSize).toBe(tSync.vocabSize);
    expect(tAsync.encode(corpus)).toEqual(tSync.encode(corpus));
  });

  it('trainOnUtf8Bytes matches train() on the same UTF-8 bytes', () => {
    const corpus = 'hello \n世界 abab';
    const cap = 340;
    const enc = new TextEncoder();
    const bytes = enc.encode(corpus);
    const tStr = new BPETokenizer();
    const tBytes = new BPETokenizer();
    tStr.train(corpus, cap);
    tBytes.trainOnUtf8Bytes(bytes, cap);
    expect(tBytes.vocabSize).toBe(tStr.vocabSize);
    expect(tBytes.encode(corpus)).toEqual(tStr.encode(corpus));
  });

  it('trainAsyncOnUtf8Bytes matches train()', async () => {
    const corpus = 'the quick brown fox';
    const cap = 320;
    const bytes = new TextEncoder().encode(corpus);
    const a = new BPETokenizer();
    const b = new BPETokenizer();
    a.train(corpus, cap);
    await b.trainAsyncOnUtf8Bytes(bytes, cap);
    expect(b.vocabSize).toBe(a.vocabSize);
    expect(b.encode(corpus)).toEqual(a.encode(corpus));
  });

  it('decode utf8 replace matches fatal for well-formed merged text', () => {
    const corpus = 'the quick brown fox\n';
    const tok = new BPETokenizer();
    tok.train(corpus, 320);
    const ids = tok.encode(corpus);
    expect(tok.decode(ids, { utf8: 'replace' })).toBe(tok.decode(ids));
  });

  it('encode progress hooks emit expand and a terminal merge event', () => {
    const corpus = 'the quick brown fox\n';
    const tok = new BPETokenizer();
    tok.train(corpus, 320);
    const events: BpeEncodeProgressEvent[] = [];
    tok.encode(corpus, {
      onProgress: (e) => {
        events.push(e);
      },
    });
    expect(events.length).toBeGreaterThanOrEqual(2);
    expect(events[0]?.phase).toBe('expand');
    const last = events[events.length - 1];
    expect(last).toMatchObject({ phase: 'merge' });
    expect(last && last.phase === 'merge' ? last.tokenCount : 0).toBeGreaterThan(0);
  });

  it('encodeAsync agrees with encode', async () => {
    const corpus = 'abababcccdd extra';
    const tok = new BPETokenizer();
    tok.train(corpus, 320);
    await expect(tok.encodeAsync(corpus)).resolves.toEqual(tok.encode(corpus));
  });

  it('determinism: identical merges and encode for two trainers', async () => {
    const corpus = 'abababcccdd';
    const cap = 310;
    const a = new BPETokenizer();
    const b = new BPETokenizer();
    a.train(corpus, cap);
    b.train(corpus, cap);
    expect(a.vocabSize).toBe(b.vocabSize);
    expect(a.encode(corpus)).toEqual(b.encode(corpus));
    expect(a.encode('abab')).toEqual(b.encode('abab'));

    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-determinism-'));
    try {
      const fa = path.join(dir, 'a.json');
      const fb = path.join(dir, 'b.json');
      await a.save(fa);
      await b.save(fb);
      const ja = JSON.parse(await readFile(fa, 'utf8')) as { merges: unknown };
      const jb = JSON.parse(await readFile(fb, 'utf8')) as { merges: unknown };
      expect(ja.merges).toEqual(jb.merges);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('save and load preserve encode/decode', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-tokenizer-test-'));
    try {
      await mkdir(dir, { recursive: true });
      const file = path.join(dir, 'bpe.json');
      const first = new BPETokenizer();
      const corpus = 'hello world\n';
      first.train(corpus, 300);
      await first.save(file);

      const second = new BPETokenizer({ specialTokens: ['<|wrong|>'] });
      await second.load(file);
      expect(second.vocabSize).toBe(first.vocabSize);
      expect(second.encode(corpus)).toEqual(first.encode(corpus));
      expect(second.decode(first.encode('world'))).toBe('world');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('load rejects wrong kind, version, and malformed vocab', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-tokenizer-bad-'));
    try {
      await mkdir(dir, { recursive: true });
      const f1 = path.join(dir, 'bad-kind.json');
      await writeFile(f1, `${JSON.stringify({ kind: 'other', version: 1, vocab: {} })}\n`);
      await expect(new BPETokenizer().load(f1)).rejects.toThrow(/expected kind/);

      const specials = ['<|pad|>', '<|endoftext|>'] as const;
      const f2 = path.join(dir, 'bad-version.json');
      await writeFile(
        f2,
        `${JSON.stringify({
          kind: 'llm-transformer-diy/bpe-tokenizer',
          version: 99,
          specialTokens: specials,
          byteOffset: specials.length,
          merges: [],
          vocab: baseVocabMap(specials),
        })}\n`,
      );
      await expect(new BPETokenizer().load(f2)).rejects.toThrow(/unsupported version/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('decode throws on unknown ids', () => {
    const tok = new BPETokenizer();
    tok.train('ab', minVocab);
    expect(() => tok.decode([9999])).toThrow(/unknown id/);
  });

  it('decode rejects id equal to vocabSize (one past last valid id)', () => {
    const tok = new BPETokenizer();
    tok.train('ab', minVocab);
    expect(() => tok.decode([tok.vocabSize])).toThrow(/unknown id/);
  });

  it('decode rejects undefined entries in ids array', () => {
    const tok = new BPETokenizer();
    tok.train('ab', minVocab);
    const first = tok.encode('a')[0];
    expect(first).toBeDefined();
    const sparse: number[] = [first];
    sparse.length = 2;
    expect(() => tok.decode(sparse)).toThrow(/undefined/);
  });

  it('training stops early when no mergeable pairs; vocabSize stays at base', () => {
    const tok = new BPETokenizer();
    tok.train('a', 400);
    expect(tok.vocabSize).toBe(minVocab);
  });

  it('load rejects byteOffset not matching specials length', async () => {
    const specials = [...DEFAULT_SPECIALS];
    const { path: filePath, cleanup } = await tmpJson({
      kind: 'llm-transformer-diy/bpe-tokenizer',
      version: 2,
      pretokenizer: 'word-and-punct-v1',
      specialTokens: specials,
      byteOffset: 1,
      merges: [],
      vocab: baseVocabMap(specials),
    });
    try {
      await expect(new BPETokenizer().load(filePath)).rejects.toThrow(/byteOffset/);
    } finally {
      await cleanup();
    }
  });

  it('load rejects missing contiguous vocab id', async () => {
    const specials = [...DEFAULT_SPECIALS];
    const vocab = vocabWithoutRow(baseVocabMap(specials), 255);
    const { path: filePath, cleanup } = await tmpJson({
      kind: 'llm-transformer-diy/bpe-tokenizer',
      version: 2,
      pretokenizer: 'word-and-punct-v1',
      specialTokens: specials,
      byteOffset: specials.length,
      merges: [],
      vocab,
    });
    try {
      await expect(new BPETokenizer().load(filePath)).rejects.toThrow(
        /missing vocab row for id 255/,
      );
    } finally {
      await cleanup();
    }
  });

  it('load rejects max vocab id inconsistent with merge count', async () => {
    const specials = [...DEFAULT_SPECIALS];
    const S = specials.length;
    const merged = oneMergeVocabWithBytesAtMergeId({
      specials,
      mergeLeft: S + 97,
      mergeRight: S + 97,
      mergedIdBytes: [97, 97],
    });
    const vocab = vocabWithoutRow(merged.vocab, S + 256);
    const { path: filePath, cleanup } = await tmpJson({
      kind: 'llm-transformer-diy/bpe-tokenizer',
      version: 2,
      pretokenizer: 'word-and-punct-v1',
      specialTokens: specials,
      byteOffset: S,
      merges: merged.merges,
      vocab,
    });
    try {
      await expect(new BPETokenizer().load(filePath)).rejects.toThrow(/expected max vocab id/);
    } finally {
      await cleanup();
    }
  });

  it('load rejects vocab byte out of 0–255', async () => {
    const specials = [...DEFAULT_SPECIALS];
    const vocab = baseVocabMap(specials);
    vocab[String(200)] = [256];
    const { path: filePath, cleanup } = await tmpJson({
      kind: 'llm-transformer-diy/bpe-tokenizer',
      version: 2,
      pretokenizer: 'word-and-punct-v1',
      specialTokens: specials,
      byteOffset: specials.length,
      merges: [],
      vocab,
    });
    try {
      await expect(new BPETokenizer().load(filePath)).rejects.toThrow(/out of 0/);
    } finally {
      await cleanup();
    }
  });

  it('load rejects merge entries that are not numeric pairs', async () => {
    const specials = [...DEFAULT_SPECIALS];
    const { path: filePath, cleanup } = await tmpJson({
      kind: 'llm-transformer-diy/bpe-tokenizer',
      version: 2,
      pretokenizer: 'word-and-punct-v1',
      specialTokens: specials,
      byteOffset: specials.length,
      merges: JSON.parse('[["x","y"]]') as [number, number][],
      vocab: baseVocabMap(specials),
    });
    try {
      await expect(new BPETokenizer().load(filePath)).rejects.toThrow(/each merge must be/);
    } finally {
      await cleanup();
    }
  });

  it('decode surfaces ill-formed UTF-8 from loaded merged row (fatal decoder)', async () => {
    const specials = [...DEFAULT_SPECIALS];
    const S = specials.length;
    const la = S + 'a'.charCodeAt(0);
    const { vocab, merges } = oneMergeVocabWithBytesAtMergeId({
      specials,
      mergeLeft: la,
      mergeRight: la,
      mergedIdBytes: [0xed, 0xa0, 0x80],
    });
    const { path: filePath, cleanup } = await tmpJson({
      kind: 'llm-transformer-diy/bpe-tokenizer',
      version: 2,
      pretokenizer: 'word-and-punct-v1',
      specialTokens: specials,
      byteOffset: S,
      merges,
      vocab,
    });
    try {
      const tok = new BPETokenizer();
      await tok.load(filePath);
      expect(tok.vocabSize).toBe(S + 256 + 1);
      expect(() => tok.decode([S + 256])).toThrow(/ill-formed UTF-8/);
      expect(tok.decode([S + 256], { utf8: 'replace' })).toContain('\uFFFD');
    } finally {
      await cleanup();
    }
  });

  it('handles empty string after training on empty corpus', () => {
    const tok = new BPETokenizer();
    tok.train('', minVocab);
    expect(tok.encode('')).toEqual([]);
    expect(tok.decode([])).toBe('');
  });
});
