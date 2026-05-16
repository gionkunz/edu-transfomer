import { readFile, writeFile } from 'node:fs/promises';

import {
  materializeFilesIntoSlotPool,
  materializeUtf8BytesIntoSlotPool,
  materializeUtf8BytesIntoSlotPoolSync,
} from './bpe-train-incremental/materialize';
import {
  buildInitialPairIndex,
  runIncrementalMergeLoop,
  runIncrementalMergeLoopSync,
} from './bpe-train-incremental/run-merge-loop';
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
      /** Outer best-rank passes completed in the **current region** (each pass collapses
       * every non-overlapping occurrence of the chosen pair).
       */
      readonly mergeIteration: number;
      /** Aggregate token count across all regions encoded so far in the call. */
      readonly tokenCount: number;
      /** Zero-based index of the region whose pass count is being reported. */
      readonly regionIndex: number;
      /** Total regions in the active encode call (known after pre-tokenization). */
      readonly regionsTotal: number;
    };

/**
 * Pre-tokenizer policy applied by **`encode`** and **`train`** before any BPE step.
 *
 * Currently a single label — **`word-and-punct-v1`** (split on whitespace / digits /
 * Unicode letters / other). The label is persisted in the tokenizer JSON so a future
 * policy can be added without breaking loaded tokenizers.
 */
export type PretokenizerPolicy = 'word-and-punct-v1';

/**
 * Regex backing **`word-and-punct-v1`**: matches contiguous runs of one of four kinds —
 * whitespace, ASCII digits, Unicode letters, or everything else (punctuation / symbols).
 * Every code point falls under exactly one alternative, so the concatenation of matches
 * is byte-for-byte equal to the original input.
 */
export const PRETOKENIZER_REGEX_WORD_AND_PUNCT_V1: RegExp = /\s+|\d+|\p{L}+|[^\s\p{L}\p{N}]+/gu;

/**
 * Split **`text`** into adjacent non-overlapping **regions** under **`policy`**.
 *
 * Each match of {@link PRETOKENIZER_REGEX_WORD_AND_PUNCT_V1} produces one region; the
 * concatenation equals **`text`**. Allocates exactly one fresh array per call.
 */
export function splitIntoRegions(text: string, _policy: PretokenizerPolicy): readonly string[] {
  if (text.length === 0) {
    return [];
  }
  const out: string[] = [];
  PRETOKENIZER_REGEX_WORD_AND_PUNCT_V1.lastIndex = 0;
  for (const m of text.matchAll(PRETOKENIZER_REGEX_WORD_AND_PUNCT_V1)) {
    out.push(m[0]);
  }
  return out;
}

/**
 * Build a region-boundary bitmap over **`corpusUtf8`** under **`policy`**, sized to the
 * byte length. **`boundary[i] === 1`** means "the pair `(byte[i], byte[i+1])` crosses a
 * region boundary and SHALL NOT be counted during BPE training".
 *
 * Walks UTF-8 bytes and emits a `1` at every byte position that is the **last byte of a
 * region** (i.e. the next byte starts a new region). Every region boundary in the JS
 * string corresponds to a code-point boundary, hence a byte boundary.
 */
function buildRegionBoundaryBitmap(
  corpusUtf8: Uint8Array,
  policy: PretokenizerPolicy,
  textForRegions: string | null,
): Uint8Array {
  if (corpusUtf8.length === 0) return new Uint8Array(0);

  const boundary = new Uint8Array(corpusUtf8.length);

  if (textForRegions !== null) {
    // Fast path: JS string already in hand (in-memory training). Use the exact regex
    // to stay perfectly consistent with `splitIntoRegions`.
    if (textForRegions.length === 0) return boundary;
    const utf8 = new TextEncoder();
    let byteCursor = 0;
    let firstRegion = true;
    PRETOKENIZER_REGEX_WORD_AND_PUNCT_V1.lastIndex = 0;
    for (const m of textForRegions.matchAll(PRETOKENIZER_REGEX_WORD_AND_PUNCT_V1)) {
      const regionBytes = utf8.encode(m[0]);
      if (!firstRegion) {
        const prevLastIdx = byteCursor - 1;
        if (prevLastIdx >= 0 && prevLastIdx < boundary.length) {
          boundary[prevLastIdx] = 1;
        }
      }
      firstRegion = false;
      byteCursor += regionBytes.length;
    }
    return boundary;
  }

  // No string available (large file-based corpus). V8 strings are capped at ~512 MB, so
  // decoding a 1+ GB byte array crashes. Classify code points directly from the raw UTF-8
  // bytes instead — equivalent to the regex for all characters the regex matches.
  fillBoundaryFromUtf8Bytes(corpusUtf8, boundary, policy);
  return boundary;
}

/**
 * Per-module cache for non-ASCII code point classification under `word-and-punct-v1`.
 * Values: 0 = whitespace, 1 = digit, 2 = letter, 3 = other.
 * The cache is populated lazily via single-char regex tests and is stable across calls.
 */
const WORD_PUNCT_CP_CLASS_CACHE = new Map<number, 0 | 1 | 2 | 3>();

/**
 * Classify a Unicode code point as one of four `word-and-punct-v1` character classes:
 * `0` whitespace (`\s`), `1` digit (`\d`), `2` letter (`\p{L}`), `3` other.
 *
 * ASCII is handled by direct comparison (fast path). Non-ASCII code points are tested
 * once via single-character regex tests and the result is cached.
 */
function classifyWordAndPunctCodePoint(cp: number): 0 | 1 | 2 | 3 {
  if (cp < 0x80) {
    // ASCII fast path — covers ≥ 95 % of typical training text.
    if (cp === 0x09 || cp === 0x0a || cp === 0x0b || cp === 0x0c || cp === 0x0d || cp === 0x20)
      return 0; // whitespace
    if (cp >= 0x30 && cp <= 0x39) return 1; // digit
    if ((cp >= 0x41 && cp <= 0x5a) || (cp >= 0x61 && cp <= 0x7a)) return 2; // letter
    return 3; // other
  }
  const cached = WORD_PUNCT_CP_CLASS_CACHE.get(cp);
  if (cached !== undefined) return cached;
  const ch = String.fromCodePoint(cp);
  let cls: 0 | 1 | 2 | 3;
  if (/^\s$/u.test(ch)) cls = 0;
  else if (/^\d$/u.test(ch)) cls = 1;
  else if (/^\p{L}$/u.test(ch)) cls = 2;
  else cls = 3;
  WORD_PUNCT_CP_CLASS_CACHE.set(cp, cls);
  return cls;
}

/**
 * Walk raw UTF-8 bytes, classify each code point under `word-and-punct-v1`,
 * and set `boundary[i] = 1` for every byte `i` that is the **last byte of a region**
 * (i.e. the next byte starts a code point of a different class).
 *
 * Ill-formed UTF-8 bytes (lone continuation bytes) are each treated as one OT token
 * so the cursor stays in sync without throwing.
 */
function fillBoundaryFromUtf8Bytes(
  bytes: Uint8Array,
  boundary: Uint8Array,
  policy: PretokenizerPolicy,
): void {
  const n = bytes.length;
  let prevCls = -1;
  let i = 0;
  while (i < n) {
    const b0 = bytes[i];
    let cp: number;
    let seqLen: number;
    if (b0 < 0x80) {
      cp = b0;
      seqLen = 1;
    } else if (b0 < 0xc2) {
      // Lone continuation byte — ill-formed UTF-8; treat as OT and advance by 1.
      cp = b0;
      seqLen = 1;
    } else if (b0 < 0xe0) {
      cp = ((b0 & 0x1f) << 6) | (i + 1 < n ? bytes[i + 1] & 0x3f : 0);
      seqLen = 2;
    } else if (b0 < 0xf0) {
      cp =
        ((b0 & 0x0f) << 12) |
        (i + 1 < n ? (bytes[i + 1] & 0x3f) << 6 : 0) |
        (i + 2 < n ? bytes[i + 2] & 0x3f : 0);
      seqLen = 3;
    } else {
      cp =
        ((b0 & 0x07) << 18) |
        (i + 1 < n ? (bytes[i + 1] & 0x3f) << 12 : 0) |
        (i + 2 < n ? (bytes[i + 2] & 0x3f) << 6 : 0) |
        (i + 3 < n ? bytes[i + 3] & 0x3f : 0);
      seqLen = 4;
    }
    const cls = classifyWordAndPunctCodePoint(cp);
    if (prevCls !== -1 && cls !== prevCls) {
      // Class transition: the last byte of the previous code point ends a region.
      boundary[i - 1] = 1;
    }
    prevCls = cls;
    i += seqLen;
  }
  void policy; // used only for error messages on the string path
}

/** Instrumentation callbacks for **`encode`** (beyond the **`Tokenizer`** contract). */
export type BpeEncodeHooks = {
  /**
   * Progress callbacks during encoding.
   *
   * For **`phase: 'merge'`**, **`mergeIteration`** counts **outer best-rank passes**: each pass picks
   * the minimum-rank applicable adjacent pair and collapses **all** of its non-overlapping
   * left-to-right occurrences in one in-place walk (not a single pair merge).
   *
   * A valid pessimistic upper bound remains **`max(1, initialTokenCount - 1)`** from the preceding
   * **`expand`** event: each pass removes at least one token slot in the worst case.
   */
  readonly onProgress?: (event: BpeEncodeProgressEvent) => void;
  /**
   * After **`encodeAsync`** / **`encodeUint32Async`** yields to the event loop (~50ms wall-clock
   * slices and/or after merge progress emit). Use for repaint / elapsed-time refresh without
   * implying a semantic merge milestone.
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

/** Emit merge **`onProgress`** during sync encode every this many outer passes. */
const EMIT_PASS_EVERY = 2048;
/**
 * Secondary progress cadence for async encode: emit every this many outer passes (primary trigger
 * is wall-time via **`TIME_SLICE_ENCODE_ASYNC_MS`**).
 */
const EMIT_PASS_EVERY_ASYNC = 8;

/** Max wall time between event-loop yields during **`encodeAsync`** (input, spinners, repaints). */
const TIME_SLICE_ENCODE_ASYNC_MS = 50;

function yieldEventLoopForHostUi(): Promise<void> {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, 0);
  });
}

const JSON_KIND = 'llm-transformer-diy/bpe-tokenizer' as const;
const JSON_VERSION = 2 as const;
const KNOWN_PRETOKENIZER_LABELS = new Set<PretokenizerPolicy>(['word-and-punct-v1']);

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

/**
 * Encode hot-path lookups for **`(L, R)`** pairs use a **single-level** `Map<number, number>`
 * keyed by the packed integer **`L * vocabSize + R`**. One **`Map.get`** per probe — no
 * intermediate row allocation, no optional chain, no nested map.
 *
 * Safe while **`vocabSize ≤ floor(sqrt(2^53))`** (about **`94 906 265`**), well above the
 * **`O(10^4–10^5)`** range used by this project. Larger vocabularies SHALL fall back to a
 * nested-map representation (not exercised here today, see
 * `tokenizer-bpe-bootstrap` spec for the conditional invariant).
 */
const PACKED_PAIR_KEY_SAFE_VOCAB_MAX = Math.floor(Math.sqrt(Number.MAX_SAFE_INTEGER));

function assertPackedKeySafe(vocabSize: number, label: string): void {
  if (vocabSize > PACKED_PAIR_KEY_SAFE_VOCAB_MAX) {
    throw new Error(
      `${label}: packed-key invariant requires vocabSize <= ${String(PACKED_PAIR_KEY_SAFE_VOCAB_MAX)} (got ${String(vocabSize)}); the nested-map fallback is not yet implemented in this build`,
    );
  }
}

/** Payload for **`onMergeComplete`** during BPE training (RAM **`train`** / **`trainAsync`** or **`trainFromFiles`**). */
export type BpeTrainMergeCompleteEvent = {
  readonly mergeIndex: number;
  readonly newVocabId: number;
  /** **`Uint32Array` backing buffer size** (**`4`** bytes per token slot) for the post-merge logical stream. */
  readonly tokenStreamBytes: bigint;
};

/** Progress / UI hooks for **`trainFromFiles`** (optional). */
export type BpeTrainFromFilesHooks = {
  /**
   * Corpus UTF-8 bytes read (**excludes** synthetic **`\\n`** between files).
   * Throttled during streaming reads; guaranteed final call at end with exact total.
   */
  readonly onCorpusUtf8Bytes?: (cumulativeUtf8Bytes: bigint) => void;

  /**
   * Fires **exactly once** per training run, immediately after the initial pair-frequency
   * table has been built from the materialized corpus and before the first merge.
   * The argument is the count of distinct adjacent pairs observed in the initial scan.
   */
  readonly onPairIndexBuilt?: (distinctInitialPairs: number) => void;

  readonly onMergeComplete?: (evt: BpeTrainMergeCompleteEvent) => void;

  /**
   * Bytes occupied by the materialized linked-list slot pool, in `u32`-equivalent
   * units (i.e. `liveCount * 4`). With incremental in-place merges, the per-merge
   * file rewrite is gone — this callback fires **once** at the end of the
   * materialization phase. Subsequent stream-size signal is conveyed via
   * `onMergeComplete.tokenStreamBytes`.
   */
  readonly onTokenStreamBytesWritten?: (cumulativeTokenFileBytes: bigint) => void;

  /** Event-loop yields for TUIs (**Ctrl+C**) / repaint — **after every merge** **and** between materialize stripes / per disk-stream chunk. */
  readonly yieldToHost?: () => void | Promise<void>;
};

/** Options for **`trainFromFiles`** (reads each file as raw UTF-8 bytes; no giant in-memory `Buffer` concat). */
export type BpeTrainFromFilesOptions = {
  readonly hooks?: BpeTrainFromFilesHooks;
};

/**
 * Hooks for in-memory **`train`** / **`trainAsync`**.
 * Prefer **`trainAsync`** when **`yieldToHost`** is async (**TUIs**, cooperative **Ctrl+C**).
 */
export type BpeTrainHooks = Pick<
  BpeTrainFromFilesHooks,
  'onMergeComplete' | 'onPairIndexBuilt' | 'yieldToHost'
>;

/** Options for **`BPETokenizer.train`** (optional **`3rd`** arg). */
export type BpeTrainOptions = {
  readonly hooks?: BpeTrainHooks;
};

interface BPETokenizerJsonParsed {
  readonly kind: typeof JSON_KIND;
  readonly version: typeof JSON_VERSION;
  readonly specialTokens: readonly string[];
  readonly byteOffset: number;
  readonly merges: readonly (readonly [number, number])[];
  readonly vocab: Readonly<Record<string, readonly number[]>>;
  readonly pretokenizer: PretokenizerPolicy;
}

export class BPETokenizer implements Tokenizer {
  private readonly specialTokens: string[];
  private readonly textEncoder = new TextEncoder();

  /** Dense rows: ids `0 .. activeVocabCount - 1` */
  private vocabByteRows: Uint8Array[] = [];
  /** Merge creation order (`mergeRank` = index) */
  private mergesRecorded: [number, number][] = [];
  /** Merge rank & output id keyed by the **packed** pair `L * vocabSize + R` — one **`Map.get`** per encode probe. */
  private mergeRankByPackedPair = new Map<number, number>();
  private mergeOutputIdByPackedPair = new Map<number, number>();

  private activeVocabCount = 0;

  /** Active pre-tokenizer policy. Set by `train{,Async,...}` / `load`; immutable on a trained tokenizer. */
  private pretokenizerPolicy: PretokenizerPolicy = 'word-and-punct-v1';

  constructor(options?: BPETokenizerOptions) {
    const defaults = ['<|pad|>', '<|endoftext|>'] as const;
    const specials = options?.specialTokens ?? defaults;
    assertValidSpecials(specials);
    this.specialTokens = [...specials];
  }

  get vocabSize(): number {
    return this.activeVocabCount;
  }

  /** Active pre-tokenizer policy label. */
  get pretokenizer(): PretokenizerPolicy {
    return this.pretokenizerPolicy;
  }

  /** Pack `(L, R)` into the single-level Map key. Caller guarantees both are `< vocabSize`. */
  private packPairKey(L: number, R: number): number {
    return L * this.activeVocabCount + R;
  }

  /**
   * Build the initial vocab byte rows: specials at ids `0 .. S-1`, then the 256
   * single-byte tokens at ids `S .. S+255`. Returns the rows and the floor `S + 256`.
   */
  private buildInitialVocabRows(): { rows: Uint8Array[]; floor: number; S: number } {
    const S = this.specialTokens.length;
    const floor = S + 256;
    const rows: Uint8Array[] = [];
    for (let i = 0; i < S; i++) {
      const tok = this.specialTokens[i];
      if (tok === undefined) throw new Error('BPETokenizer.train: specials hole');
      rows[i] = this.textEncoder.encode(tok);
    }
    for (let b = 0; b < 256; b++) {
      rows[S + b] = new Uint8Array([b]);
    }
    return { rows, floor, S };
  }

  private assertVocabCapAtLeastFloor(
    vocabSizeRequested: number,
    floor: number,
    label: string,
  ): void {
    if (!Number.isFinite(vocabSizeRequested) || vocabSizeRequested < floor) {
      throw new Error(
        `${label}: vocabSize ${String(vocabSizeRequested)} is less than required minimum ${String(floor)} (specialTokens=${String(this.specialTokens.length)} + 256 byte tokens)`,
      );
    }
  }

  /** Synchronous RAM training (**`Tokenizer.train`** shape + optional **`3rd`**-arg hooks). */
  train(text: string, vocabSizeRequested: number, options?: BpeTrainOptions): void {
    const bytes = this.textEncoder.encode(text);
    this.runRamTrainingSync(bytes, vocabSizeRequested, options, text);
  }

  /** Byte-level training on UTF-8 bytes (same semantics as **`train(chunksAsString)`**). */
  trainOnUtf8Bytes(
    corpusUtf8: Uint8Array,
    vocabSizeRequested: number,
    options?: BpeTrainOptions,
  ): void {
    this.runRamTrainingSync(corpusUtf8, vocabSizeRequested, options, null);
  }

  /**
   * Async RAM training on raw UTF-8 bytes (avoids a giant JS string + duplicate **`encode`** buffer).
   * Yields **after each materialize stripe** **and after every merge** (**`yieldToHost`**) so stdin /
   * **`SIGINT`** can run during long merge passes.
   */
  async trainAsyncOnUtf8Bytes(
    corpusUtf8: Uint8Array,
    vocabSizeRequested: number,
    options?: BpeTrainOptions,
  ): Promise<void> {
    await this.runRamTrainingAsync(corpusUtf8, vocabSizeRequested, options, null);
  }

  /**
   * Async RAM training: same merges as **`train`**, plus **`yieldToHost`** **after materialize stripes**
   * **and after every merge** (**`train(text)`** still encodes UTF-8 in one shot up front).
   */
  async trainAsync(
    text: string,
    vocabSizeRequested: number,
    options?: BpeTrainOptions,
  ): Promise<void> {
    const bytes = this.textEncoder.encode(text);
    await this.runRamTrainingAsync(bytes, vocabSizeRequested, options, text);
  }

  /** Sync RAM training core (no microtask hops). Backs `train` and `trainOnUtf8Bytes`. */
  private runRamTrainingSync(
    corpusUtf8: Uint8Array,
    vocabSizeRequested: number,
    options: BpeTrainOptions | undefined,
    textForRegions: string | null,
  ): void {
    const { rows, floor, S } = this.buildInitialVocabRows();
    this.assertVocabCapAtLeastFloor(vocabSizeRequested, floor, 'BPETokenizer.train');

    const policy: PretokenizerPolicy = 'word-and-punct-v1';
    const regionBoundary = buildRegionBoundaryBitmap(corpusUtf8, policy, textForRegions);

    const stream = materializeUtf8BytesIntoSlotPoolSync(corpusUtf8, S);
    const { index, occList, distinctInitialPairs } = buildInitialPairIndex(stream, regionBoundary);
    options?.hooks?.onPairIndexBuilt?.(distinctInitialPairs);

    const result = runIncrementalMergeLoopSync({
      stream,
      index,
      occList,
      rows,
      vocabCap: vocabSizeRequested,
      nextIdInitial: floor,
      hooks: {
        ...(options?.hooks?.onMergeComplete !== undefined
          ? { onMergeComplete: options.hooks.onMergeComplete }
          : {}),
      },
      regionBoundary,
    });

    this.finalizeBpeTraining(rows, result.mergesList, result.nextId, floor, policy);
  }

  /** Async RAM training core. Backs `trainAsync` and `trainAsyncOnUtf8Bytes`. */
  private async runRamTrainingAsync(
    corpusUtf8: Uint8Array,
    vocabSizeRequested: number,
    options: BpeTrainOptions | undefined,
    textForRegions: string | null,
  ): Promise<void> {
    const { rows, floor, S } = this.buildInitialVocabRows();
    this.assertVocabCapAtLeastFloor(vocabSizeRequested, floor, 'BPETokenizer.train');

    const policy: PretokenizerPolicy = 'word-and-punct-v1';
    const regionBoundary = buildRegionBoundaryBitmap(corpusUtf8, policy, textForRegions);

    const stream = await materializeUtf8BytesIntoSlotPool(corpusUtf8, S, {
      ...(options?.hooks?.yieldToHost !== undefined
        ? { yieldToHost: options.hooks.yieldToHost }
        : {}),
    });
    const { index, occList, distinctInitialPairs } = buildInitialPairIndex(stream, regionBoundary);
    options?.hooks?.onPairIndexBuilt?.(distinctInitialPairs);

    const result = await runIncrementalMergeLoop({
      stream,
      index,
      occList,
      rows,
      vocabCap: vocabSizeRequested,
      nextIdInitial: floor,
      hooks: {
        ...(options?.hooks?.onMergeComplete !== undefined
          ? { onMergeComplete: options.hooks.onMergeComplete }
          : {}),
        ...(options?.hooks?.yieldToHost !== undefined
          ? { yieldToHost: options.hooks.yieldToHost }
          : {}),
      },
      regionBoundary,
    });

    this.finalizeBpeTraining(rows, result.mergesList, result.nextId, floor, policy);
  }

  /**
   * Byte-level BPE training on many files **without** loading the full corpus as one JS string.
   *
   * Reads each path as **raw UTF-8 bytes** (same as training on joined file contents with **`\n`**
   * between files), materializes a single in-memory linked-list slot pool sized exactly to the
   * total byte length, then runs the incremental merge loop directly against that pool. The legacy
   * "round-trip a `u32` LE token stream through disk between merges" model is gone — every merge is
   * an `O(K_m + P_m)` in-place operation against the slot pool and no temp files are created.
   */
  async trainFromFiles(
    sortedAbsolutePaths: readonly string[],
    vocabSizeRequested: number,
    options?: BpeTrainFromFilesOptions,
  ): Promise<void> {
    if (sortedAbsolutePaths.length === 0) {
      throw new Error('BPETokenizer.trainFromFiles: at least one file path is required');
    }

    const { rows, floor, S } = this.buildInitialVocabRows();
    this.assertVocabCapAtLeastFloor(vocabSizeRequested, floor, 'BPETokenizer.trainFromFiles');

    const policy: PretokenizerPolicy = 'word-and-punct-v1';

    const stream = await materializeFilesIntoSlotPool(sortedAbsolutePaths, S, {
      ...(options?.hooks?.onCorpusUtf8Bytes !== undefined
        ? { onCorpusUtf8Bytes: options.hooks.onCorpusUtf8Bytes }
        : {}),
      ...(options?.hooks?.onTokenStreamBytesWritten !== undefined
        ? { onTokenStreamBytesWritten: options.hooks.onTokenStreamBytesWritten }
        : {}),
      ...(options?.hooks?.yieldToHost !== undefined
        ? { yieldToHost: options.hooks.yieldToHost }
        : {}),
    });
    if (options?.hooks?.yieldToHost !== undefined) {
      await options.hooks.yieldToHost();
    }

    // For pre-tokenization we need the raw corpus bytes to find region boundaries.
    // The streaming materializer has already encoded each byte into a slot at id
    // `S + byte`; recover the bytes by walking the slot pool's `value` array (the
    // slots are populated in append order, so indices `0 .. liveCount-1` carry the
    // original byte stream including the synthetic `\n` separators between files).
    const regionBoundary = this.buildRegionBoundaryFromSlotStream(stream, S);

    const { index, occList, distinctInitialPairs } = buildInitialPairIndex(stream, regionBoundary);
    options?.hooks?.onPairIndexBuilt?.(distinctInitialPairs);

    const result = await runIncrementalMergeLoop({
      stream,
      index,
      occList,
      rows,
      vocabCap: vocabSizeRequested,
      nextIdInitial: floor,
      hooks: {
        ...(options?.hooks?.onMergeComplete !== undefined
          ? { onMergeComplete: options.hooks.onMergeComplete }
          : {}),
        ...(options?.hooks?.yieldToHost !== undefined
          ? { yieldToHost: options.hooks.yieldToHost }
          : {}),
      },
      regionBoundary,
    });

    this.finalizeBpeTraining(rows, result.mergesList, result.nextId, floor, policy);
  }

  /**
   * Recover the raw UTF-8 byte stream from a freshly-materialized slot pool and run the
   * pre-tokenizer regex over it. Returns a region-boundary bitmap sized to `stream.capacity`.
   *
   * Only safe to call **before any merge** (so `value[i] - S` is exactly the original byte).
   */
  private buildRegionBoundaryFromSlotStream(
    stream: { value: Uint32Array; capacity: number; liveCount: number },
    S: number,
  ): Uint8Array {
    const n = stream.liveCount;
    if (n === 0) return new Uint8Array(stream.capacity);
    const bytes = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      bytes[i] = stream.value[i] - S;
    }
    const boundaryNarrow = buildRegionBoundaryBitmap(bytes, 'word-and-punct-v1', null);
    const boundaryFull = new Uint8Array(stream.capacity);
    boundaryFull.set(boundaryNarrow.subarray(0, n));
    return boundaryFull;
  }

  private finalizeBpeTraining(
    rows: Uint8Array[],
    mergesList: [number, number][],
    nextId: number,
    floor: number,
    pretokenizerPolicy: PretokenizerPolicy,
  ): void {
    this.vocabByteRows = rows;
    this.mergesRecorded = mergesList;
    this.activeVocabCount = nextId;
    this.pretokenizerPolicy = pretokenizerPolicy;
    this.rebuildPackedPairTables(floor);
  }

  /**
   * Build the encode-hot-path packed-pair tables (`mergeRankByPackedPair`,
   * `mergeOutputIdByPackedPair`) from `mergesRecorded` and `activeVocabCount`.
   * Called by `finalizeBpeTraining` and `load`. Keys are `L * activeVocabCount + R`.
   */
  private rebuildPackedPairTables(floor: number): void {
    assertPackedKeySafe(this.activeVocabCount, 'BPETokenizer');
    const rankMap = new Map<number, number>();
    const idMap = new Map<number, number>();
    let mergedIdWalker = floor;
    for (let rank = 0; rank < this.mergesRecorded.length; rank++) {
      const pair = this.mergesRecorded[rank];
      if (pair === undefined) throw new Error('BPETokenizer: merge hole');
      const Lm = pair[0];
      const Rm = pair[1];
      if (Lm >= this.activeVocabCount || Rm >= this.activeVocabCount) {
        throw new Error(
          `BPETokenizer: merge rank ${String(rank)} references id >= vocabSize (${String(Lm)},${String(Rm)} vs ${String(this.activeVocabCount)})`,
        );
      }
      const key = Lm * this.activeVocabCount + Rm;
      idMap.set(key, mergedIdWalker);
      rankMap.set(key, rank);
      mergedIdWalker += 1;
    }
    this.mergeRankByPackedPair = rankMap;
    this.mergeOutputIdByPackedPair = idMap;
    if (mergedIdWalker !== this.activeVocabCount) {
      throw new Error('BPETokenizer: internal merge counter mismatch');
    }
  }

  /**
   * One greedy ranked **BPE** pass on **`buf[0..len)`**: pick the minimum-rank applicable pair
   * (tie-break: smaller **`L`**, then smaller **`R`**), then replace all non-overlapping
   * left-to-right occurrences in a single read/write walk. Mutates **`buf`** in place; returns
   * the new logical length, or **`null`** when no learned pair applies.
   */
  private performGreedyMergePassUint32(buf: Uint32Array, len: number): number | null {
    if (len < 2) return null;

    const V = this.activeVocabCount;
    const rankByPacked = this.mergeRankByPackedPair;
    const idByPacked = this.mergeOutputIdByPackedPair;

    let minRank = Infinity;
    let chosenL = Infinity;
    let chosenR = Infinity;

    for (let j = 0; j < len - 1; j++) {
      const L = buf[j];
      const R = buf[j + 1];
      const rk = rankByPacked.get(L * V + R);
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

    const mergedId = idByPacked.get(chosenL * V + chosenR);
    if (mergedId === undefined) {
      return null;
    }

    let w = 0;
    let r = 0;
    while (r < len) {
      if (r + 1 < len && buf[r] === chosenL && buf[r + 1] === chosenR) {
        buf[w] = mergedId;
        w += 1;
        r += 2;
      } else {
        buf[w] = buf[r];
        w += 1;
        r += 1;
      }
    }
    return w;
  }

  /**
   * Expand a single region's UTF-8 bytes into byte-seed token ids on top of a shared
   * working buffer at offset 0. Returns the initial token length for this region.
   */
  private expandRegionIntoBuffer(buf: Uint32Array, regionBytes: Uint8Array, S: number): number {
    const n = regionBytes.length;
    for (let i = 0; i < n; i++) {
      buf[i] = S + regionBytes[i];
    }
    return n;
  }

  /**
   * Encode to a freshly allocated **`Uint32Array`** (length = logical token count; not a
   * subarray view into shared working storage).
   *
   * Each pre-tokenizer region is encoded independently; merges never cross region boundaries.
   * The implementation works directly on UTF-8 bytes to avoid the O(regions) string-object
   * overhead that `splitIntoRegions` would cause for corpora with millions of regions.
   *
   * CPU-bound; progress hooks run but TUIs/timer paints stall until this returns.
   */
  encodeUint32(text: string, hooks?: BpeEncodeHooks): Uint32Array {
    this.assertTrained();
    const emit = hooks?.onProgress;
    const S = this.specialTokens.length;

    // Single UTF-8 encode of the full text.
    const allBytes = this.textEncoder.encode(text);
    const B = allBytes.length;

    emit?.({
      phase: 'expand',
      textChars: text.length,
      byteLength: B,
      initialTokenCount: B,
    });

    if (B === 0) return new Uint32Array(0);

    // Build boundary bitmap via byte-level classifier — O(B), no per-region string allocs.
    const boundary = new Uint8Array(B);
    fillBoundaryFromUtf8Bytes(allBytes, boundary, this.pretokenizerPolicy);

    // Count regions for progress emission (single O(B) scan).
    let regionsTotal = 1;
    for (let i = 0; i < B - 1; i++) {
      if (boundary[i] === 1) regionsTotal++;
    }

    // Output buffer: start at B/4 (assumes ≥4x compression, typical for a trained BPE vocab)
    // and double on demand. This avoids allocating 4× the byte count up-front.
    let outCapacity = Math.max(65536, B >> 2);
    let outputBuf = new Uint32Array(outCapacity);
    let outputLen = 0;
    // Reusable working buffer sized to the largest region we encounter; grows lazily.
    let workBuf = new Uint32Array(0);

    let regionStart = 0;
    for (let r = 0; r < regionsTotal; r++) {
      // Scan to find the exclusive end of this region.
      let regionEnd = regionStart;
      while (regionEnd < B - 1 && boundary[regionEnd] !== 1) regionEnd++;
      regionEnd++;

      const regionLen = regionEnd - regionStart;
      if (workBuf.length < regionLen) workBuf = new Uint32Array(regionLen);

      // Expand bytes → token ids directly from the byte subarray (zero-copy view).
      let len = this.expandRegionIntoBuffer(workBuf, allBytes.subarray(regionStart, regionEnd), S);

      let mergeIteration = 0;
      while (true) {
        const nextLen = this.performGreedyMergePassUint32(workBuf, len);
        if (nextLen === null) break;
        len = nextLen;
        mergeIteration += 1;
        if (
          emit !== undefined &&
          (mergeIteration % EMIT_PASS_EVERY === 0 || len <= 2 || mergeIteration <= 4)
        ) {
          emit({
            phase: 'merge',
            mergeIteration,
            tokenCount: outputLen + len,
            regionIndex: r,
            regionsTotal,
          });
        }
      }

      // Grow output buffer if needed (rare — only when compression is worse than initial guess).
      if (outputLen + len > outCapacity) {
        outCapacity = Math.min(B, Math.max(outCapacity * 2, outputLen + len));
        const grown = new Uint32Array(outCapacity);
        grown.set(outputBuf.subarray(0, outputLen));
        outputBuf = grown;
      }

      // Write region tokens directly into the output buffer — no per-region Uint32Array alloc.
      outputBuf.set(workBuf.subarray(0, len), outputLen);
      outputLen += len;

      emit?.({
        phase: 'merge',
        mergeIteration,
        tokenCount: outputLen,
        regionIndex: r,
        regionsTotal,
      });

      regionStart = regionEnd;
    }

    return outputBuf.slice(0, outputLen);
  }

  /**
   * Async **`encodeUint32`**: yields on the same wall-time / progress cadence as **`encodeAsync`**
   * so hosts stay responsive during long encodes. Folds over pre-tokenized regions.
   *
   * Uses the same byte-range strategy as the sync path — no per-region string allocations.
   */
  async encodeUint32Async(text: string, hooks?: BpeEncodeHooks): Promise<Uint32Array> {
    this.assertTrained();
    const emit = hooks?.onProgress;
    const onYieldCb = hooks?.onYield;
    const sliceYield = async (): Promise<void> => {
      await yieldEventLoopForHostUi();
      onYieldCb?.();
    };
    const S = this.specialTokens.length;

    const allBytes = this.textEncoder.encode(text);
    const B = allBytes.length;

    emit?.({
      phase: 'expand',
      textChars: text.length,
      byteLength: B,
      initialTokenCount: B,
    });
    await sliceYield();

    if (B === 0) return new Uint32Array(0);

    const boundary = new Uint8Array(B);
    fillBoundaryFromUtf8Bytes(allBytes, boundary, this.pretokenizerPolicy);

    let regionsTotal = 1;
    for (let i = 0; i < B - 1; i++) {
      if (boundary[i] === 1) regionsTotal++;
    }

    let outCapacity = Math.max(65536, B >> 2);
    let outputBuf = new Uint32Array(outCapacity);
    let outputLen = 0;
    let workBuf = new Uint32Array(0);
    let lastYieldWallMs = Date.now();

    let regionStart = 0;
    for (let r = 0; r < regionsTotal; r++) {
      let regionEnd = regionStart;
      while (regionEnd < B - 1 && boundary[regionEnd] !== 1) regionEnd++;
      regionEnd++;

      const regionLen = regionEnd - regionStart;
      if (workBuf.length < regionLen) workBuf = new Uint32Array(regionLen);

      let len = this.expandRegionIntoBuffer(workBuf, allBytes.subarray(regionStart, regionEnd), S);

      let mergeIteration = 0;
      while (true) {
        const nextLen = this.performGreedyMergePassUint32(workBuf, len);
        if (nextLen === null) break;
        len = nextLen;
        mergeIteration += 1;

        if (
          emit !== undefined &&
          (mergeIteration % EMIT_PASS_EVERY_ASYNC === 0 || len <= 2 || mergeIteration <= 4)
        ) {
          emit({
            phase: 'merge',
            mergeIteration,
            tokenCount: outputLen + len,
            regionIndex: r,
            regionsTotal,
          });
        }

        // Yield only on the wall-clock slice — not on every emit. For corpora with millions
        // of tiny regions every emit would otherwise schedule a ~1ms setTimeout, turning a
        // second of BPE work into hours of event-loop overhead.
        const nowMs = Date.now();
        if (nowMs - lastYieldWallMs >= TIME_SLICE_ENCODE_ASYNC_MS) {
          await sliceYield();
          lastYieldWallMs = Date.now();
        }
      }

      if (outputLen + len > outCapacity) {
        outCapacity = Math.min(B, Math.max(outCapacity * 2, outputLen + len));
        const grown = new Uint32Array(outCapacity);
        grown.set(outputBuf.subarray(0, outputLen));
        outputBuf = grown;
      }

      outputBuf.set(workBuf.subarray(0, len), outputLen);
      outputLen += len;

      emit?.({
        phase: 'merge',
        mergeIteration,
        tokenCount: outputLen,
        regionIndex: r,
        regionsTotal,
      });

      const nowMs = Date.now();
      if (nowMs - lastYieldWallMs >= TIME_SLICE_ENCODE_ASYNC_MS) {
        await sliceYield();
        lastYieldWallMs = Date.now();
      }

      regionStart = regionEnd;
    }

    await sliceYield();
    return outputBuf.slice(0, outputLen);
  }

  /** CPU-bound; progress hooks run but TUIs/timer paints stall until **`encode`** returns. */
  encode(text: string, hooks?: BpeEncodeHooks): number[] {
    return Array.from(this.encodeUint32(text, hooks));
  }

  /**
   * Same as **`encode`**, but periodically yields (~50ms wall clock and after merge progress)
   * so hosts can repaint and handle **`Ctrl+C`** while merges run.
   */
  async encodeAsync(text: string, hooks?: BpeEncodeHooks): Promise<number[]> {
    return Array.from(await this.encodeUint32Async(text, hooks));
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

    const payload = {
      kind: JSON_KIND,
      version: JSON_VERSION,
      pretokenizer: this.pretokenizerPolicy,
      specialTokens: this.specialTokens,
      byteOffset: S,
      merges: [...this.mergesRecorded.map((step) => [step[0], step[1]] as const)],
      vocab: vocabOut,
    } as const;

    await writeFile(path, `${JSON.stringify(payload)}\n`, 'utf8');
  }

  async load(path: string): Promise<void> {
    const rawText = await readFile(path, 'utf8');
    const parsed: unknown = JSON.parse(rawText) as unknown;
    const data = parseBpeTokenizerJson(parsed);
    this.pretokenizerPolicy = data.pretokenizer;

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
    this.activeVocabCount = maxIdSeen + 1;
    this.rebuildPackedPairTables(S + 256);
  }

  private assertTrained(): void {
    if (this.activeVocabCount === 0) {
      throw new Error('BPETokenizer: call train() or load() before encode/decode/save');
    }
  }
}

function parseBpeTokenizerJson(raw: unknown): BPETokenizerJsonParsed {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('BPETokenizer.load: root must be a JSON object');
  }
  const rec = raw as Record<string, unknown>;
  if (rec.kind !== JSON_KIND) {
    throw new Error(
      `BPETokenizer.load: expected kind ${JSON.stringify(JSON_KIND)}, got ${JSON.stringify(rec.kind)}`,
    );
  }
  if (rec.version !== JSON_VERSION) {
    throw new Error(
      `BPETokenizer.load: unsupported version ${JSON.stringify(rec.version)} (need ${String(JSON_VERSION)}); retrain the tokenizer`,
    );
  }
  const version = rec.version;

  if (!('pretokenizer' in rec)) {
    throw new Error(
      `BPETokenizer.load: missing required "pretokenizer" field (supported: ${[...KNOWN_PRETOKENIZER_LABELS].map((k) => JSON.stringify(k)).join(', ')})`,
    );
  }
  const rawPolicy = rec.pretokenizer;
  if (
    typeof rawPolicy !== 'string' ||
    !KNOWN_PRETOKENIZER_LABELS.has(rawPolicy as PretokenizerPolicy)
  ) {
    throw new Error(
      `BPETokenizer.load: unknown pretokenizer ${JSON.stringify(rawPolicy)} (supported: ${[...KNOWN_PRETOKENIZER_LABELS].map((k) => JSON.stringify(k)).join(', ')})`,
    );
  }
  const pretokenizer: PretokenizerPolicy = rawPolicy as PretokenizerPolicy;

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
    version,
    pretokenizer,
    specialTokens,
    byteOffset: rec.byteOffset,
    merges,
    vocab,
  };
}
