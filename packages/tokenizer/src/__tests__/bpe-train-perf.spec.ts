import { performance } from 'node:perf_hooks';

import { describe, expect, it } from 'vitest';

import { bpePairKey, pickBestMergePair } from '../bpe-merge-stats';
import { BPETokenizer, splitIntoRegions } from '../bpe-tokenizer';

/**
 * Perf comparison between the new incremental training path and the legacy
 * full-rescan + greedy-rewrite reference. Skipped by default in CI so it
 * doesn't add minutes to the green-light loop; opt in with `RUN_BPE_PERF=1`.
 *
 * Run via:
 *   RUN_BPE_PERF=1 npx nx run @llm-transformer-diy/tokenizer:test \
 *     -- --run src/__tests__/bpe-train-perf.spec.ts
 *
 * Records before/after wall time + merges/sec; the assertion at the end is
 * an extremely lenient correctness check (identical merge sequence) so the
 * test never fails for performance reasons — its primary job is to surface
 * numbers that get pasted into `train-throughput-note.md`.
 */

const ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789 .,';

function generateSyntheticCorpus(seed: number, byteCount: number): string {
  // Deterministic LCG over a tiny alphabet — same RNG state means same input
  // bytes for the new and reference paths so we can compare them apples-to-apples.
  let state = (seed | 0) >>> 0;
  const out: string[] = [];
  const alphaLen = ALPHABET.length;
  for (let i = 0; i < byteCount; i++) {
    state = (Math.imul(state, 1_103_515_245) + 12_345) >>> 0;
    const ch = ALPHABET[state % alphaLen];
    if (ch === undefined) throw new Error('unreachable: ALPHABET index out of range');
    out.push(ch);
  }
  return out.join('');
}

interface ReferenceTrainResult {
  readonly merges: [number, number][];
  readonly elapsedMs: number;
}

function trainReferenceFullRescan(
  corpus: string,
  vocabSize: number,
  specials: readonly string[],
): ReferenceTrainResult {
  const enc = new TextEncoder();
  const S = specials.length;
  const floor = S + 256;
  let regions: Uint32Array[] = splitIntoRegions(corpus, 'word-and-punct-v1').map((r) => {
    const b = enc.encode(r);
    const u = new Uint32Array(b.length);
    for (let i = 0; i < b.length; i++) u[i] = S + b[i];
    return u;
  });
  const merges: [number, number][] = [];
  const start = performance.now();
  let nextId = floor;
  while (nextId < vocabSize) {
    // Reference baseline — full O(N) rescan per merge across regions — kept here only so the
    // benchmark has something concrete to measure the incremental path against.
    const freq = new Map<string, number>();
    for (const region of regions) {
      for (let i = 0; i + 1 < region.length; i++) {
        const L = region[i];
        const R = region[i + 1];
        const key = bpePairKey(L, R);
        freq.set(key, (freq.get(key) ?? 0) + 1);
      }
    }
    const pair = pickBestMergePair(freq);
    if (pair === null) break;
    merges.push([pair.L, pair.R]);
    const next: Uint32Array[] = [];
    for (const region of regions) {
      let outLen = 0;
      let i = 0;
      while (i < region.length) {
        if (i + 1 < region.length && region[i] === pair.L && region[i + 1] === pair.R) {
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
      while (i < region.length) {
        if (i + 1 < region.length && region[i] === pair.L && region[i + 1] === pair.R) {
          out[w++] = nextId;
          i += 2;
        } else {
          out[w++] = region[i];
          i += 1;
        }
      }
      next.push(out);
    }
    regions = next;
    nextId += 1;
  }
  return { merges, elapsedMs: performance.now() - start };
}

const RUN = process.env.RUN_BPE_PERF === '1';

/**
 * Configurable workload sizes. The defaults are tuned so the reference (which is
 * `O(N · M)` on a `Map<string, number>`) finishes in roughly a minute on a
 * developer laptop. Override with env vars to scale up:
 *
 *   RUN_BPE_PERF=1 BPE_PERF_BYTES=1048576 BPE_PERF_MERGES=512 \
 *     npx nx run @llm-transformer-diy/tokenizer:test \
 *     -- --run src/__tests__/bpe-train-perf.spec.ts
 */
function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`${name} must be a positive integer (got ${JSON.stringify(raw)})`);
  }
  return n;
}

describe.skipIf(!RUN)('BPE training perf: incremental vs full-rescan reference', () => {
  it(
    'produces identical merge sequences and reports wall-time + merges/sec',
    async () => {
      const SEED = 0x12_34_56_78;
      const BYTES = envInt('BPE_PERF_BYTES', 512 * 1024); // 512 KiB default
      const MERGES = envInt('BPE_PERF_MERGES', 256);
      const VOCAB_SIZE = 2 + 256 + MERGES;
      const corpus = generateSyntheticCorpus(SEED, BYTES);

      console.log(
        `[BPE perf] starting reference (full-rescan, O(N·M)) — corpus=${String(BYTES)} bytes, target merges=${String(MERGES)}…`,
      );
      const ref = trainReferenceFullRescan(corpus, VOCAB_SIZE, ['<|pad|>', '<|endoftext|>']);
      console.log(
        `[BPE perf] reference done in ${ref.elapsedMs.toFixed(1)} ms (${String(ref.merges.length)} merges)`,
      );

      const tok = new BPETokenizer();
      const startNew = performance.now();
      await tok.trainAsyncOnUtf8Bytes(new TextEncoder().encode(corpus), VOCAB_SIZE);
      const newElapsedMs = performance.now() - startNew;

      const newMergesCount = tok.vocabSize - (2 + 256);
      expect(newMergesCount).toBe(ref.merges.length);

      const refMergesPerSec = (ref.merges.length / ref.elapsedMs) * 1000;
      const newMergesPerSec = (newMergesCount / newElapsedMs) * 1000;

      // Print numbers; do not assert beyond the merge-count parity above so the
      // spec never fails for hardware reasons.
      console.log(
        [
          `[BPE perf] corpus=${String(BYTES)} bytes, vocabSize=${String(VOCAB_SIZE)} (${String(ref.merges.length)} merges)`,
          `[BPE perf] reference (full-rescan): ${ref.elapsedMs.toFixed(1)} ms (${refMergesPerSec.toFixed(1)} merges/sec)`,
          `[BPE perf] incremental:             ${newElapsedMs.toFixed(1)} ms (${newMergesPerSec.toFixed(1)} merges/sec)`,
          `[BPE perf] speedup:                 ${(ref.elapsedMs / newElapsedMs).toFixed(2)}×`,
        ].join('\n'),
      );
    },
    30 * 60 * 1000,
  );
});
