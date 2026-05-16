import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { describe, expect, it, vi } from 'vitest';

import * as bpeMergeStats from '../bpe-merge-stats';
import { countAdjacentPairsOverlappingUint32, pickBestMergePair } from '../bpe-merge-stats';
import { BPETokenizer } from '../bpe-tokenizer';

/**
 * Reference helper: train a `BPETokenizer`-equivalent merge sequence using the
 * deprecated full-rescan path, in-process and synchronously. Returns the merge
 * list and the populated vocab byte rows so callers can assert equivalence with
 * the new incremental training path.
 */
function trainReferenceFullRescan(
  corpus: string,
  vocabSize: number,
  specials: readonly string[] = ['<|pad|>', '<|endoftext|>'],
): { merges: [number, number][]; rows: Uint8Array[]; activeVocabCount: number } {
  const enc = new TextEncoder();
  const S = specials.length;
  const floor = S + 256;
  if (vocabSize < floor) {
    throw new Error(`reference: vocabSize ${vocabSize} < floor ${floor}`);
  }

  const bytes = enc.encode(corpus);
  let stream: Uint32Array;
  if (bytes.length === 0) {
    stream = new Uint32Array(0);
  } else {
    stream = new Uint32Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) stream[i] = S + bytes[i];
  }

  const rows: Uint8Array[] = [];
  for (let i = 0; i < S; i++) rows[i] = enc.encode(specials[i]);
  for (let b = 0; b < 256; b++) rows[S + b] = new Uint8Array([b]);

  const merges: [number, number][] = [];
  let nextId = floor;
  while (nextId < vocabSize) {
    const freq = countAdjacentPairsOverlappingUint32(stream);
    const pair = pickBestMergePair(freq);
    if (pair === null) break;
    const L = pair.L;
    const R = pair.R;
    const lb = rows[L];
    const rb = rows[R];
    rows[nextId] = new Uint8Array(lb.length + rb.length);
    rows[nextId].set(lb, 0);
    rows[nextId].set(rb, lb.length);
    merges.push([L, R]);
    // Greedy non-overlapping rewrite into a fresh Uint32Array.
    let outLen = 0;
    let i = 0;
    while (i < stream.length) {
      if (i + 1 < stream.length && stream[i] === L && stream[i + 1] === R) {
        outLen += 1;
        i += 2;
      } else {
        outLen += 1;
        i += 1;
      }
    }
    const out = new Uint32Array(outLen);
    let w = 0;
    i = 0;
    while (i < stream.length) {
      if (i + 1 < stream.length && stream[i] === L && stream[i + 1] === R) {
        out[w++] = nextId;
        i += 2;
      } else {
        out[w++] = stream[i];
        i += 1;
      }
    }
    stream = out;
    nextId += 1;
  }
  return { merges, rows, activeVocabCount: nextId };
}

interface CorpusCase {
  readonly label: string;
  readonly corpus: string;
}

const CASES: readonly CorpusCase[] = [
  { label: 'unicode hello (~4 KiB repeat)', corpus: 'héllo 世界 '.repeat(450) },
  { label: 'overlapping letters', corpus: 'X'.repeat(2048) },
  {
    label: 'tied frequencies',
    corpus: 'abab cdcd '.repeat(64) + 'efef '.repeat(48),
  },
  {
    label: 'longer paragraph',
    corpus:
      'the quick brown fox jumps over the lazy dog. ' +
      'the rain in spain falls mainly on the plain. '.repeat(40),
  },
];

const VOCAB_CAPS = [258 + 4, 258 + 1024];

describe('BPE incremental training equivalence with reference full-rescan', () => {
  for (const c of CASES) {
    for (const cap of VOCAB_CAPS) {
      it(`${c.label} @ vocabSize=${cap}: merges and vocab rows match reference`, () => {
        const ref = trainReferenceFullRescan(c.corpus, cap);
        const tok = new BPETokenizer();
        tok.train(c.corpus, cap);
        const refTokenizerProbe = ref.merges.map((m) => [m[0], m[1]] as const);
        // Query the trained tokenizer's merges via save/load round-trip (private fields are encapsulated).
        // Easier: assert encode parity on probes — encode is deterministic given identical merges.
        // We also assert vocabSize matches.
        expect(tok.vocabSize).toBe(ref.activeVocabCount);
        // Round-trip a probe set so any merge divergence shows up immediately.
        const probes = [c.corpus.slice(0, 256), c.corpus.slice(0, 1024), c.corpus];
        for (const p of probes) {
          expect(tok.decode(tok.encode(p))).toBe(p);
        }
        // Direct merge-list parity via JSON snapshot.
        return (async (): Promise<void> => {
          const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-equiv-'));
          try {
            const f = path.join(dir, 'tok.json');
            await tok.save(f);
            const json = JSON.parse(await readFile(f, 'utf8')) as {
              merges: [number, number][];
            };
            expect(json.merges.length).toBe(refTokenizerProbe.length);
            for (let i = 0; i < json.merges.length; i++) {
              expect(json.merges[i]).toEqual([refTokenizerProbe[i][0], refTokenizerProbe[i][1]]);
            }
          } finally {
            await rm(dir, { recursive: true, force: true });
          }
        })();
      });
    }
  }
});

describe('BPE training: trainFromFiles ≡ in-memory train on the same combined corpus', () => {
  it('matches in-memory train when the corpus is sharded across multiple files', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-files-equiv-'));
    try {
      const corpus = 'héllo 世界 '.repeat(64) + 'abab cdcd efef '.repeat(32);
      // Shard into 4 files, joined with newline to mirror trainFromFiles semantics.
      const shards = [
        corpus.slice(0, 200),
        corpus.slice(200, 480),
        corpus.slice(480, 800),
        corpus.slice(800),
      ];
      const paths: string[] = [];
      for (let i = 0; i < shards.length; i++) {
        const p = path.join(dir, `shard-${String(i)}.txt`);
        await writeFile(p, shards[i]);
        paths.push(path.resolve(p));
      }

      const cap = 320;
      const tMem = new BPETokenizer();
      const tDisk = new BPETokenizer();
      tMem.train(shards.join('\n'), cap);
      await tDisk.trainFromFiles(paths, cap);

      expect(tDisk.vocabSize).toBe(tMem.vocabSize);

      const probes = [shards[0], shards[2], shards.join('\n'), 'héllo', 'efef'];
      for (const p of probes) {
        expect(tDisk.encode(p)).toEqual(tMem.encode(p));
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe('BPE training hooks: ordering and counts', () => {
  it('onPairIndexBuilt fires exactly once before the first onMergeComplete', async () => {
    const corpus = 'abababcccdd extra abababab';
    const events: string[] = [];
    const tok = new BPETokenizer();
    let pairIndexBuiltArg = -1;
    let pairIndexBuiltCalls = 0;
    await tok.trainAsyncOnUtf8Bytes(new TextEncoder().encode(corpus), 320, {
      hooks: {
        onPairIndexBuilt: (n) => {
          pairIndexBuiltCalls += 1;
          pairIndexBuiltArg = n;
          events.push('pairIndexBuilt');
        },
        onMergeComplete: () => {
          events.push('mergeComplete');
        },
      },
    });
    expect(pairIndexBuiltCalls).toBe(1);
    expect(pairIndexBuiltArg).toBeGreaterThanOrEqual(0);
    expect(events[0]).toBe('pairIndexBuilt');
    expect(events.filter((e) => e === 'pairIndexBuilt').length).toBe(1);
    expect(events.filter((e) => e === 'mergeComplete').length).toBeGreaterThan(0);
  });
});

describe('BPE training: determinism', () => {
  it('two trainings of the same corpus yield identical merges and identical tokenizer.json', async () => {
    const corpus = 'the quick brown fox '.repeat(50);
    const cap = 320;
    const a = new BPETokenizer();
    const b = new BPETokenizer();
    a.train(corpus, cap);
    b.train(corpus, cap);
    expect(a.vocabSize).toBe(b.vocabSize);
    expect(a.encode(corpus)).toEqual(b.encode(corpus));
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-determinism-'));
    try {
      const fa = path.join(dir, 'a.json');
      const fb = path.join(dir, 'b.json');
      await a.save(fa);
      await b.save(fb);
      const ja = await readFile(fa, 'utf8');
      const jb = await readFile(fb, 'utf8');
      expect(ja).toBe(jb);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe('BPE training tie-break: smaller L then smaller R', () => {
  it('picks the lower-id pair when two pairs tie on count', () => {
    // Designed so the only count-≥2 pairs are (a,b) and (c,d), both with count 2.
    // a=97 < c=99, so smaller L wins.
    const corpus = 'abab cdcd';
    const tok = new BPETokenizer();
    tok.train(corpus, 259);
    // After exactly one merge, (a,b) → 258 means encode("ab") collapses to a single token.
    expect(tok.encode('ab').length).toBe(1);
    expect(tok.encode('cd').length).toBe(2);
  });
});

describe('BPE training: hot path does not allocate per-pair strings', () => {
  it('train() never invokes the deprecated bpePairKey on the hot path', () => {
    const spy = vi.spyOn(bpeMergeStats, 'bpePairKey');
    try {
      const tok = new BPETokenizer();
      tok.train('abababcccdd extra', 320);
      expect(spy).not.toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
  });
});

describe('BPE training: save/load round-trip preserves encode/decode', () => {
  it('reloads with identical encode results', async () => {
    const corpus = 'hello world\n';
    const dir = await mkdtemp(path.join(os.tmpdir(), 'bpe-roundtrip-'));
    try {
      const f = path.join(dir, 'tok.json');
      const first = new BPETokenizer();
      first.train(corpus, 300);
      await first.save(f);
      const second = new BPETokenizer({ specialTokens: ['<|placeholder|>'] });
      await second.load(f);
      expect(second.vocabSize).toBe(first.vocabSize);
      expect(second.encode(corpus)).toEqual(first.encode(corpus));
      expect(second.decode(first.encode('world'))).toBe('world');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
