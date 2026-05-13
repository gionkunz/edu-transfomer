import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { BPETokenizer, type BpeEncodeProgressEvent } from '../bpe-tokenizer';

type PairMap = Map<number, Map<number, number>>;

function nestedGet(m: PairMap, a: number, b: number): number | undefined {
  return m.get(a)?.get(b);
}

/** One merge: min-rank pair, **leftmost** occurrence only (reference for equivalence). */
function performSingleGreedyMergeLeftmost(
  mergeRankByPair: PairMap,
  mergeOutputIdByPair: PairMap,
  buf: Uint32Array,
  len: number,
): number | null {
  if (len < 2) return null;

  let minRank = Infinity;
  let chosenL = Infinity;
  let chosenR = Infinity;

  for (let j = 0; j < len - 1; j++) {
    const L = buf[j];
    const R = buf[j + 1];
    const rk = nestedGet(mergeRankByPair, L, R);
    if (rk === undefined) continue;
    if (rk < minRank || (rk === minRank && (L < chosenL || (L === chosenL && R < chosenR)))) {
      minRank = rk;
      chosenL = L;
      chosenR = R;
    }
  }

  if (minRank === Infinity) return null;

  let jAt = -1;
  for (let k = 0; k < len - 1; k++) {
    if (buf[k] === chosenL && buf[k + 1] === chosenR) {
      jAt = k;
      break;
    }
  }
  if (jAt < 0) return null;

  const mergedId = nestedGet(mergeOutputIdByPair, chosenL, chosenR);
  if (mergedId === undefined) return null;

  buf[jAt] = mergedId;
  if (jAt + 2 < len) {
    buf.copyWithin(jAt + 1, jAt + 2, len);
  }
  return len - 1;
}

type BpeMergeTables = {
  mergeRankByPair: PairMap;
  mergeOutputIdByPair: PairMap;
  specialTokens: string[];
  textEncoder: TextEncoder;
};

function encodeLeftmostOnlyReference(tok: BPETokenizer, text: string): number[] {
  const { mergeRankByPair, mergeOutputIdByPair, specialTokens, textEncoder } =
    tok as unknown as BpeMergeTables;
  const S = specialTokens.length;
  const bytes = textEncoder.encode(text);
  const n = bytes.length;
  const buf = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    buf[i] = S + bytes[i];
  }
  let len = n;
  while (true) {
    const nextLen = performSingleGreedyMergeLeftmost(
      mergeRankByPair,
      mergeOutputIdByPair,
      buf,
      len,
    );
    if (nextLen === null) break;
    len = nextLen;
  }
  return Array.from(buf.subarray(0, len));
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixtureParagraph = path.join(__dirname, '../../../../corpus/test/pg35_trim2.txt');

describe('BPE encode hot loop (merge-all pass vs leftmost baseline)', () => {
  it('matches leftmost-only reference on Unicode sample, repeating digram, and fixture paragraph', async () => {
    const unicodeSample = 'héllo 世界';
    const tokA = new BPETokenizer();
    tokA.train(unicodeSample, 400);
    const baseA = encodeLeftmostOnlyReference(tokA, unicodeSample);
    expect(tokA.encode(unicodeSample)).toEqual(baseA);
    expect(Array.from(tokA.encodeUint32(unicodeSample))).toEqual(baseA);

    const repeatCorpus = 'abababab';
    const tokB = new BPETokenizer();
    tokB.train(repeatCorpus, 320);
    const baseB = encodeLeftmostOnlyReference(tokB, repeatCorpus);
    expect(tokB.encode(repeatCorpus)).toEqual(baseB);
    expect(Array.from(tokB.encodeUint32(repeatCorpus))).toEqual(baseB);

    const rawFixture = await readFile(fixtureParagraph, 'utf8');
    const paragraph = rawFixture.slice(0, 12_000);
    const tokC = new BPETokenizer();
    tokC.train(paragraph, 320);
    const baseC = encodeLeftmostOnlyReference(tokC, paragraph);
    expect(tokC.encode(paragraph)).toEqual(baseC);
    expect(Array.from(tokC.encodeUint32(paragraph))).toEqual(baseC);
  }, 60_000);

  it('encodeUint32 matches encode on standard probes and determinism corpus', () => {
    const probes = ['héllo 世界\nping 123', 'the quick brown fox\n', 'abababcccdd extra', 'abab'];
    for (const corpus of probes) {
      const tok = new BPETokenizer();
      tok.train(corpus, 320);
      expect(Array.from(tok.encodeUint32(corpus))).toEqual(tok.encode(corpus));
    }

    const corpus = 'abababcccdd';
    const cap = 310;
    const a = new BPETokenizer();
    const b = new BPETokenizer();
    a.train(corpus, cap);
    b.train(corpus, cap);
    expect(Array.from(a.encodeUint32(corpus))).toEqual(a.encode(corpus));
    expect(Array.from(b.encodeUint32(corpus))).toEqual(b.encode(corpus));
  });

  it('encodeUint32Async matches encodeAsync', async () => {
    const probes = ['abababcccdd extra', 'the quick brown fox\n', 'héllo 世界'];
    for (const corpus of probes) {
      const tok = new BPETokenizer();
      tok.train(corpus, 320);
      const sync = await tok.encodeAsync(corpus);
      const u = await tok.encodeUint32Async(corpus);
      expect(Array.from(u)).toEqual(sync);
    }
  });

  it('mergeIteration from encodeUint32Async is monotonic and within pessimistic bound', async () => {
    const corpus = 'x'.repeat(4000);
    const tok = new BPETokenizer();
    tok.train(corpus, 400);

    let initialTokenCount = 0;
    const mergeIters: number[] = [];
    await tok.encodeUint32Async(corpus, {
      onProgress: (ev: BpeEncodeProgressEvent) => {
        if (ev.phase === 'expand') {
          initialTokenCount = ev.initialTokenCount;
        } else if (ev.phase === 'merge') {
          mergeIters.push(ev.mergeIteration);
        }
      },
    });

    expect(initialTokenCount).toBeGreaterThan(1);
    const upper = Math.max(1, initialTokenCount - 1);
    for (let i = 1; i < mergeIters.length; i++) {
      expect(mergeIters[i]).toBeGreaterThanOrEqual(mergeIters[i - 1]);
    }
    const last = mergeIters[mergeIters.length - 1];
    expect(last).toBeDefined();
    expect(last).toBeLessThanOrEqual(upper);
  });
});
