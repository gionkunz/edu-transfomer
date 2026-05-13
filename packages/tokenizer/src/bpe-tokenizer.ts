import { readFile, writeFile } from 'node:fs/promises';

import type { Tokenizer } from './types';

export interface BPETokenizerOptions {
  /**
   * Assigned ids `0 .. length - 1` in order (UTF-8 byte sequences backing each id).
   * Same constraints as CharTokenizer defaults.
   */
  readonly specialTokens?: readonly string[];
}

export type BpeEncodeProgressEvent =
  | {
      readonly phase: 'expand';
      readonly textChars: number;
      readonly byteLength: number;
      readonly initialTokenCount: number;
    }
  | {
      readonly phase: 'merge';
      readonly mergeIteration: number;
      readonly tokenCount: number;
    };

/** Instrumentation callbacks for **`encode`** (beyond the **`Tokenizer`** contract). */
export type BpeEncodeHooks = {
  readonly onProgress?: (event: BpeEncodeProgressEvent) => void;
  /**
   * After **`encodeAsync`** yields to the event loop (~50ms slices or after a merge progress emit).
   * Use for repaint / elapsed-time refresh without implying a semantic merge milestone.
   */
  readonly onYield?: () => void;
};

/** Optional **`BPETokenizer.decode`** behavior (narrow **`Tokenizer`** contract remains single‑arg **`decode`**). */
export type BpeDecodeOptions = {
  /**
   * - **`fatal`** (default): ill‑formed UTF‑8 raises (training / strict tooling).
   * - **`replace`**: **`U+FFFD`** substitutions — weak models sometimes sample invalid UTF‑8 merges.
   */
  readonly utf8?: 'fatal' | 'replace';
};

const EMIT_MERGE_EVERY = 2048;
/** Progress hook cadence for **`encodeAsync`** (host repaints more often than sync **`encode`**). */
const EMIT_MERGE_EVERY_ASYNC = 512;

/** Max wall time between event-loop yields during **`encodeAsync`** (input, spinners, repaints). */
const TIME_SLICE_ENCODE_ASYNC_MS = 50;

function yieldEventLoopForHostUi(): Promise<void> {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, 0);
  });
}

const JSON_KIND = 'llm-transformer-diy/bpe-tokenizer' as const;
const JSON_VERSION = 1 as const;

const fatalDecoder = new TextDecoder('utf-8', { fatal: true });

/** Loose UTF‑8 (**U+FFFD** substitutions) — for inference when sampled ids concatenate to invalid sequences. */
const replacingUtf8Decoder = new TextDecoder('utf-8', { fatal: false });

function assertValidSpecials(specials: readonly string[]): void {
  if (specials.length === 0) {
    throw new Error('BPETokenizer: specialTokens must be non-empty');
  }
  const seen = new Set<string>();
  for (const s of specials) {
    if (s.length === 0) {
      throw new Error('BPETokenizer: specialTokens must not contain empty strings');
    }
    if (seen.has(s)) {
      throw new Error(`BPETokenizer: duplicate special token ${JSON.stringify(s)}`);
    }
    seen.add(s);
  }
}

function pairKey(L: number, R: number): string {
  return `${String(L)}\0${String(R)}`;
}

/** Merge lookup tables use **`outer.get(L).get(R)`** so **`encode`** does not allocate a string per adjacent pair probe. */
type PairNestedNumberMap = Map<number, Map<number, number>>;

function nestedNumberGet(m: PairNestedNumberMap, a: number, b: number): number | undefined {
  return m.get(a)?.get(b);
}

function nestedNumberSet(m: PairNestedNumberMap, a: number, b: number, v: number): void {
  let row = m.get(a);
  if (row === undefined) {
    row = new Map();
    m.set(a, row);
  }
  row.set(b, v);
}

function concatByteSeq(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

interface BPETokenizerJsonV1 {
  readonly kind: typeof JSON_KIND;
  readonly version: typeof JSON_VERSION;
  readonly specialTokens: readonly string[];
  readonly byteOffset: number;
  readonly merges: readonly (readonly [number, number])[];
  readonly vocab: Readonly<Record<string, readonly number[]>>;
}

export class BPETokenizer implements Tokenizer {
  private readonly specialTokens: string[];
  private readonly textEncoder = new TextEncoder();

  /** Dense rows: ids `0 .. activeVocabCount - 1` */
  private vocabByteRows: Uint8Array[] = [];
  /** Merge creation order (`mergeRank` = index) */
  private mergesRecorded: [number, number][] = [];
  /** Merge rank & output id keyed by **`(L,R)`** (nested maps — fast **`encode`** probes). */
  private mergeRankByPair = new Map<number, Map<number, number>>();
  private mergeOutputIdByPair = new Map<number, Map<number, number>>();

  private activeVocabCount = 0;

  constructor(options?: BPETokenizerOptions) {
    const defaults = ['<|pad|>', '<|endoftext|>'] as const;
    const specials = options?.specialTokens ?? defaults;
    assertValidSpecials(specials);
    this.specialTokens = [...specials];
  }

  get vocabSize(): number {
    return this.activeVocabCount;
  }

  train(text: string, vocabSizeRequested: number): void {
    const S = this.specialTokens.length;
    const floor = S + 256;
    if (!Number.isFinite(vocabSizeRequested) || vocabSizeRequested < floor) {
      throw new Error(
        `BPETokenizer.train: vocabSize ${String(vocabSizeRequested)} is less than required minimum ${String(floor)} (specialTokens=${String(S)} + 256 byte tokens)`,
      );
    }

    const bytes = this.textEncoder.encode(text);
    let stream = bytes.length === 0 ? new Uint32Array(0) : Uint32Array.from(bytes, (b) => S + b);

    const rows: Uint8Array[] = [];
    for (let i = 0; i < S; i++) {
      const tok = this.specialTokens[i];
      if (tok === undefined) throw new Error('BPETokenizer.train: specials hole');
      rows[i] = this.textEncoder.encode(tok);
    }
    for (let b = 0; b < 256; b++) {
      rows[S + b] = new Uint8Array([b]);
    }

    let nextId = floor;
    const mergesList: [number, number][] = [];

    while (nextId < vocabSizeRequested) {
      const freq = countAdjacentPairsOverlapping(stream);

      let bestFreq = -1;
      let bestL = Infinity;
      let bestR = Infinity;
      let foundPair = false;

      for (const [k, c] of freq) {
        if (c < 2) continue;
        const ix = k.indexOf('\0');
        const L = Number(k.slice(0, ix));
        const R = Number(k.slice(ix + 1));
        const better =
          !foundPair ||
          c > bestFreq ||
          (c === bestFreq && (L < bestL || (L === bestL && R < bestR)));

        if (better) {
          foundPair = true;
          bestFreq = c;
          bestL = L;
          bestR = R;
        }
      }

      if (!foundPair || bestL === Infinity || bestR === Infinity) {
        break;
      }

      const L = bestL;
      const R = bestR;
      const lb = rows[L];
      const rb = rows[R];
      if (lb === undefined || rb === undefined) {
        throw new Error(
          `BPETokenizer.train: internal vocab hole for ids ${String(L)}, ${String(R)}`,
        );
      }

      rows[nextId] = concatByteSeq(lb, rb);
      mergesList.push([L, R]);
      stream = rewriteGreedy(stream, L, R, nextId);
      nextId += 1;
    }

    this.vocabByteRows = rows;
    this.mergesRecorded = mergesList;

    const pairToMergedNum = new Map<number, Map<number, number>>();
    const pairRankNum = new Map<number, Map<number, number>>();
    let mergedIdWalker = floor;
    for (let rank = 0; rank < mergesList.length; rank++) {
      const pair = mergesList[rank];
      if (pair === undefined) throw new Error('BPETokenizer.train: merge hole');
      const Lm = pair[0];
      const Rm = pair[1];
      nestedNumberSet(pairToMergedNum, Lm, Rm, mergedIdWalker);
      nestedNumberSet(pairRankNum, Lm, Rm, rank);
      mergedIdWalker += 1;
    }
    this.mergeOutputIdByPair = pairToMergedNum;
    this.mergeRankByPair = pairRankNum;

    this.activeVocabCount = nextId;
  }

  /**
   * One greedy ranked merge (**BPE**) step on a dense prefix **`buf[0..len)`**.
   * Mutates **`buf`** in-place; returns new logical length or **`null`** if no merge occurred.
   */
  private performSingleGreedyMergeUint32(buf: Uint32Array, len: number): number | null {
    if (len < 2) return null;

    let minRank = Infinity;
    let chosenL = Infinity;
    let chosenR = Infinity;

    for (let j = 0; j < len - 1; j++) {
      const L = buf[j];
      const R = buf[j + 1];
      const rk = nestedNumberGet(this.mergeRankByPair, L, R);
      if (rk === undefined) continue;

      if (rk < minRank || (rk === minRank && (L < chosenL || (L === chosenL && R < chosenR)))) {
        minRank = rk;
        chosenL = L;
        chosenR = R;
      }
    }

    if (minRank === Infinity) {
      return null;
    }

    let jAt = -1;
    for (let k = 0; k < len - 1; k++) {
      if (buf[k] === chosenL && buf[k + 1] === chosenR) {
        jAt = k;
        break;
      }
    }
    if (jAt < 0) {
      return null;
    }

    const mergedId = nestedNumberGet(this.mergeOutputIdByPair, chosenL, chosenR);
    if (mergedId === undefined) {
      return null;
    }

    buf[jAt] = mergedId;
    if (jAt + 2 < len) {
      buf.copyWithin(jAt + 1, jAt + 2, len);
    }
    return len - 1;
  }

  /** CPU-bound; progress hooks run but TUIs/timer paints stall until **`encode`** returns. */
  encode(text: string, hooks?: BpeEncodeHooks): number[] {
    this.assertTrained();
    const emit = hooks?.onProgress;
    const S = this.specialTokens.length;
    const bytes = this.textEncoder.encode(text);
    const n = bytes.length;
    const buf = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      buf[i] = S + bytes[i];
    }
    let len = n;

    emit?.({
      phase: 'expand',
      textChars: text.length,
      byteLength: bytes.length,
      initialTokenCount: len,
    });

    let mergeIteration = 0;
    while (true) {
      const nextLen = this.performSingleGreedyMergeUint32(buf, len);
      if (nextLen === null) break;
      len = nextLen;
      mergeIteration += 1;
      if (
        emit !== undefined &&
        (mergeIteration % EMIT_MERGE_EVERY === 0 || len <= 2 || mergeIteration <= 4)
      ) {
        emit({ phase: 'merge', mergeIteration, tokenCount: len });
      }
    }

    emit?.({
      phase: 'merge',
      mergeIteration,
      tokenCount: len,
    });

    return Array.from(buf.subarray(0, len));
  }

  /**
   * Same as **`encode`**, but periodically yields (~50ms wall clock and after merge progress)
   * so hosts can repaint and handle **`Ctrl+C`** while merges run.
   */
  async encodeAsync(text: string, hooks?: BpeEncodeHooks): Promise<number[]> {
    this.assertTrained();
    const emit = hooks?.onProgress;
    const onYieldCb = hooks?.onYield;
    const sliceYield = async (): Promise<void> => {
      await yieldEventLoopForHostUi();
      onYieldCb?.();
    };
    const S = this.specialTokens.length;
    const bytes = this.textEncoder.encode(text);
    const n = bytes.length;
    const buf = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      buf[i] = S + bytes[i];
    }
    let len = n;

    emit?.({
      phase: 'expand',
      textChars: text.length,
      byteLength: bytes.length,
      initialTokenCount: len,
    });
    await sliceYield();

    let mergeIteration = 0;
    let lastYieldWallMs = Date.now();
    while (true) {
      const nextLen = this.performSingleGreedyMergeUint32(buf, len);
      if (nextLen === null) break;
      len = nextLen;
      mergeIteration += 1;
      let emittedThisRound = false;
      if (
        emit !== undefined &&
        (mergeIteration % EMIT_MERGE_EVERY_ASYNC === 0 || len <= 2 || mergeIteration <= 4)
      ) {
        emit({ phase: 'merge', mergeIteration, tokenCount: len });
        emittedThisRound = true;
      }

      const nowMs = Date.now();
      if (emittedThisRound || nowMs - lastYieldWallMs >= TIME_SLICE_ENCODE_ASYNC_MS) {
        await sliceYield();
        lastYieldWallMs = Date.now();
      }
    }

    emit?.({
      phase: 'merge',
      mergeIteration,
      tokenCount: len,
    });
    await sliceYield();

    return Array.from(buf.subarray(0, len));
  }

  decode(ids: number[], options?: BpeDecodeOptions): string {
    this.assertTrained();
    let totalLen = 0;
    const rows: Uint8Array[] = [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (id === undefined) {
        throw new Error('BPETokenizer.decode: ids entry is undefined');
      }
      const row = this.vocabByteRows[id];
      if (id < 0 || id >= this.activeVocabCount || row === undefined) {
        throw new Error(
          `BPETokenizer.decode: unknown id ${String(id)} (vocabSize=${String(this.vocabSize)})`,
        );
      }
      rows.push(row);
      totalLen += row.length;
    }

    const all = new Uint8Array(totalLen);
    let offset = 0;
    for (const r of rows) {
      all.set(r, offset);
      offset += r.length;
    }

    if (options?.utf8 === 'replace') {
      return replacingUtf8Decoder.decode(all);
    }

    try {
      return fatalDecoder.decode(all);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(`BPETokenizer.decode: ill-formed UTF-8 byte sequence (${msg})`);
    }
  }

  async save(path: string): Promise<void> {
    this.assertTrained();
    const S = this.specialTokens.length;

    const vocabOut: Record<string, number[]> = {};
    for (let id = 0; id < this.activeVocabCount; id++) {
      const row = this.vocabByteRows[id];
      if (row === undefined) {
        throw new Error(`BPETokenizer.save: missing vocab bytes for id ${String(id)}`);
      }
      vocabOut[String(id)] = Array.from(row);
    }

    const payload: BPETokenizerJsonV1 = {
      kind: JSON_KIND,
      version: JSON_VERSION,
      specialTokens: this.specialTokens,
      byteOffset: S,
      merges: [...this.mergesRecorded.map((step) => [step[0], step[1]] as const)],
      vocab: vocabOut,
    };

    await writeFile(path, `${JSON.stringify(payload)}\n`, 'utf8');
  }

  async load(path: string): Promise<void> {
    const rawText = await readFile(path, 'utf8');
    const parsed: unknown = JSON.parse(rawText) as unknown;
    const data = parseBpeTokenizerJson(parsed);

    if (data.byteOffset !== data.specialTokens.length) {
      throw new Error(
        `BPETokenizer.load: byteOffset ${String(data.byteOffset)} disagrees with specialTokens.length ${String(data.specialTokens.length)}`,
      );
    }

    assertValidSpecials(data.specialTokens);

    const S = data.specialTokens.length;
    const expectedTopId = S + 256 + data.merges.length - 1;
    const keyNums = Object.keys(data.vocab)
      .map((k) => Number(k))
      .filter((n) => Number.isSafeInteger(n) && n >= 0);

    if (keyNums.length === 0) {
      throw new Error('BPETokenizer.load: vocab object is empty');
    }

    const maxIdSeen = Math.max(...keyNums);
    if (maxIdSeen !== expectedTopId) {
      throw new Error(
        `BPETokenizer.load: expected max vocab id ${String(expectedTopId)} (specials + 256 merges - 1); got ${String(maxIdSeen)}`,
      );
    }

    const nextRows = new Array<Uint8Array>(maxIdSeen + 1);
    for (let id = 0; id <= maxIdSeen; id++) {
      const entryUnknown = data.vocab[String(id)];
      if (!Array.isArray(entryUnknown)) {
        throw new Error(`BPETokenizer.load: missing vocab row for id ${String(id)}`);
      }
      const safeBytes: number[] = [];
      for (const el of entryUnknown) {
        if (typeof el !== 'number' || !Number.isFinite(el)) {
          throw new Error(`BPETokenizer.load: vocab byte bad type for id ${String(id)}`);
        }
        if (!Number.isInteger(el) || el < 0 || el > 255) {
          throw new Error(`BPETokenizer.load: byte value out of 0–255 for id ${String(id)}`);
        }
        safeBytes.push(el);
      }
      nextRows[id] = Uint8Array.from(safeBytes);
    }

    this.specialTokens.length = 0;
    this.specialTokens.push(...data.specialTokens);

    this.vocabByteRows = nextRows;
    this.mergesRecorded = data.merges.map((pair) => [pair[0], pair[1]]);

    const pairToMergedNum = new Map<number, Map<number, number>>();
    const pairRankNum = new Map<number, Map<number, number>>();
    let mergedIdWalker = S + 256;

    for (let rank = 0; rank < this.mergesRecorded.length; rank++) {
      const step = this.mergesRecorded[rank];
      if (step === undefined) throw new Error('BPETokenizer.load: merge hole');
      const Lm = step[0];
      const Rm = step[1];
      nestedNumberSet(pairToMergedNum, Lm, Rm, mergedIdWalker);
      nestedNumberSet(pairRankNum, Lm, Rm, rank);
      mergedIdWalker += 1;
    }

    this.mergeOutputIdByPair = pairToMergedNum;
    this.mergeRankByPair = pairRankNum;
    this.activeVocabCount = maxIdSeen + 1;

    if (mergedIdWalker !== this.activeVocabCount) {
      throw new Error('BPETokenizer.load: internal merge counter mismatch');
    }
  }

  private assertTrained(): void {
    if (this.activeVocabCount === 0) {
      throw new Error('BPETokenizer: call train() or load() before encode/decode/save');
    }
  }
}

function countAdjacentPairsOverlapping(stream: Uint32Array): Map<string, number> {
  const freq = new Map<string, number>();
  for (let i = 0; i < stream.length - 1; i++) {
    const L = stream[i];
    const R = stream[i + 1];
    if (L === undefined || R === undefined) continue;
    const pk = pairKey(L, R);
    freq.set(pk, (freq.get(pk) ?? 0) + 1);
  }
  return freq;
}

function rewriteGreedy(stream: Uint32Array, L: number, R: number, newId: number): Uint32Array {
  const tmp: number[] = [];
  let i = 0;
  while (i < stream.length) {
    const cur = stream[i];
    if (cur === undefined) {
      throw new Error('BPETokenizer.rewriteGreedy: index out of range');
    }
    if (i + 1 < stream.length && stream[i] === L && stream[i + 1] === R) {
      tmp.push(newId);
      i += 2;
    } else {
      tmp.push(cur);
      i += 1;
    }
  }
  return Uint32Array.from(tmp);
}

function parseBpeTokenizerJson(raw: unknown): BPETokenizerJsonV1 {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('BPETokenizer.load: root must be a JSON object');
  }
  const rec = raw as Record<string, unknown>;
  if (rec.kind !== JSON_KIND) {
    throw new Error(
      `BPETokenizer.load: expected kind ${JSON.stringify(JSON_KIND)}, got ${JSON.stringify(rec.kind)}`,
    );
  }
  if (typeof rec.version !== 'number' || rec.version !== JSON_VERSION) {
    throw new Error(
      `BPETokenizer.load: unsupported version ${JSON.stringify(rec.version)} (need ${String(JSON_VERSION)})`,
    );
  }
  if (!Array.isArray(rec.specialTokens)) {
    throw new Error('BPETokenizer.load: specialTokens must be string[]');
  }
  const specialTokens: string[] = [];
  for (const tok of rec.specialTokens) {
    if (typeof tok !== 'string') {
      throw new Error('BPETokenizer.load: specialTokens entries must be strings');
    }
    specialTokens.push(tok);
  }

  if (typeof rec.byteOffset !== 'number') {
    throw new Error('BPETokenizer.load: byteOffset must be a number');
  }
  if (!Array.isArray(rec.merges)) {
    throw new Error('BPETokenizer.load: merges must be an array');
  }
  const merges: [number, number][] = [];
  for (const m of rec.merges as unknown[]) {
    if (
      !Array.isArray(m) ||
      m.length !== 2 ||
      typeof m[0] !== 'number' ||
      typeof m[1] !== 'number'
    ) {
      throw new Error('BPETokenizer.load: each merge must be [number, number]');
    }
    merges.push([m[0], m[1]]);
  }

  const vocabUnknown = rec.vocab;
  if (vocabUnknown === null || typeof vocabUnknown !== 'object' || Array.isArray(vocabUnknown)) {
    throw new Error('BPETokenizer.load: vocab must be an object');
  }
  const vocRec = vocabUnknown as Record<string, unknown>;
  const vocab: Record<string, number[]> = {};

  for (const key of Object.keys(vocRec)) {
    const v = vocRec[key];
    if (!Array.isArray(v)) {
      throw new Error(`BPETokenizer.load: vocab.${key} must be number[]`);
    }
    const arr: number[] = [];
    for (const byte of v) {
      if (typeof byte !== 'number' || !Number.isFinite(byte)) {
        throw new Error(`BPETokenizer.load: vocab byte not a number (${JSON.stringify(key)})`);
      }
      if (byte < 0 || byte > 255 || !Number.isInteger(byte)) {
        throw new Error(`BPETokenizer.load: byte value out of 0–255 for id ${JSON.stringify(key)}`);
      }
      arr.push(byte);
    }
    vocab[key] = arr;
  }

  assertValidSpecials(specialTokens);

  return {
    kind: JSON_KIND,
    version: JSON_VERSION,
    specialTokens,
    byteOffset: rec.byteOffset,
    merges,
    vocab,
  };
}
