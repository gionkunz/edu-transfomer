/**
 * Corpus → linked-token-stream materializers for the incremental BPE training path.
 *
 * Two entry points:
 *
 * - {@link materializeUtf8BytesIntoSlotPool} — synchronous-friendly RAM path: takes
 *   an already-decoded `Uint8Array` and walks it in `~2 MiB` stripes, awaiting
 *   `yieldToHost?.()` between stripes so TUIs can repaint and SIGINT can fire.
 * - {@link materializeFilesIntoSlotPool} — disk-stream path: opens each input file
 *   as a `createReadStream`, walks each delivered `Buffer` chunk synchronously,
 *   inserts a synthetic UTF-8 `\n` (byte `10`) between successive files, and
 *   awaits `yieldToHost?.()` at most once per chunk (NOT per byte).
 *
 * Both paths emit one slot per UTF-8 byte at id `S + b`, matching the legacy
 * `BPETokenizer.train` byte-level seed behavior. Throws when the corpus exceeds
 * {@link MAX_SLOT_CAPACITY} (~4 GiB byte-tokens).
 */

import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';

import {
  MAX_SLOT_CAPACITY,
  appendLiveSlot,
  createLinkedTokenStream,
  type BpeLinkedTokenStream,
} from './linked-token-stream';

/** Stripe size for the RAM materializer's cooperative yield cadence. */
const RAM_MATERIALIZE_STRIPE_BYTES = 2_000_000;

/** Bytes-since-last-emit threshold for `onCorpusUtf8Bytes` during disk streaming. */
const CORPUS_PROGRESS_STRIDE_BYTES = 256 * 1024;

/** UTF-8 newline byte used as the inter-file separator (matches legacy `trainFromFiles`). */
const FILE_SEPARATOR_BYTE = 10;

/** Hooks consumed by the RAM materializer. Mirrors a subset of `BpeTrainHooks`. */
export interface RamMaterializerHooks {
  readonly yieldToHost?: () => void | Promise<void>;
}

/** Hooks consumed by the disk-stream materializer. Mirrors a subset of
 * `BpeTrainFromFilesHooks`; defined here to avoid a circular import. */
export interface DiskMaterializerHooks {
  /** Throttled (~256 KiB stride) while reading corpus UTF-8 (excludes the synthetic
   * `\n` separators between files). Final call is the exact total. */
  readonly onCorpusUtf8Bytes?: (cumulativeUtf8Bytes: bigint) => void;
  /** Single emit at end of materialization with the logical `u32`-equivalent
   * byte size of the slot pool's live tokens. */
  readonly onTokenStreamBytesWritten?: (cumulativeTokenFileBytes: bigint) => void;
  readonly yieldToHost?: () => void | Promise<void>;
}

/**
 * Synchronous variant: walk the entire byte buffer in one shot with no yields.
 * Used by the sync `BPETokenizer.train` / `trainOnUtf8Bytes` entry points so
 * those methods stay synchronous (no microtask hop between materialize and
 * encode-readiness).
 */
export function materializeUtf8BytesIntoSlotPoolSync(
  corpusUtf8: Uint8Array,
  S: number,
): BpeLinkedTokenStream {
  if (corpusUtf8.length > MAX_SLOT_CAPACITY) {
    throw new Error(
      `materializeUtf8BytesIntoSlotPoolSync: corpus byte length ${String(corpusUtf8.length)} exceeds MAX_SLOT_CAPACITY ${String(MAX_SLOT_CAPACITY)}`,
    );
  }
  const stream = createLinkedTokenStream(corpusUtf8.length);
  for (let j = 0; j < corpusUtf8.length; j++) {
    appendLiveSlot(stream, j, S + corpusUtf8[j]);
  }
  return stream;
}

/**
 * Materialize an already-decoded UTF-8 byte buffer into a fresh slot pool.
 * Each input byte `b` becomes one slot with token id `S + b`. Yields between
 * `RAM_MATERIALIZE_STRIPE_BYTES`-sized stripes when `yieldToHost` is provided.
 */
export async function materializeUtf8BytesIntoSlotPool(
  corpusUtf8: Uint8Array,
  S: number,
  hooks?: RamMaterializerHooks,
): Promise<BpeLinkedTokenStream> {
  if (corpusUtf8.length > MAX_SLOT_CAPACITY) {
    throw new Error(
      `materializeUtf8BytesIntoSlotPool: corpus byte length ${String(corpusUtf8.length)} exceeds MAX_SLOT_CAPACITY ${String(MAX_SLOT_CAPACITY)}`,
    );
  }
  const stream = createLinkedTokenStream(corpusUtf8.length);
  const yieldFn = hooks?.yieldToHost;
  let j = 0;
  while (j < corpusUtf8.length) {
    const end = Math.min(j + RAM_MATERIALIZE_STRIPE_BYTES, corpusUtf8.length);
    for (; j < end; j++) {
      appendLiveSlot(stream, j, S + corpusUtf8[j]);
    }
    if (yieldFn !== undefined) {
      await yieldFn();
    }
  }
  return stream;
}

/**
 * Materialize one or more files (joined with a synthetic UTF-8 `\n` byte between
 * successive files, mirroring legacy `trainFromFiles`) into a fresh slot pool.
 * The slot pool is sized exactly via a single `stat()` pass over the input set,
 * so no growth ever occurs during streaming.
 *
 * Cooperative yields fire at most once per delivered `Buffer` chunk — the inner
 * byte loop is synchronous and tight. Progress callbacks (`onCorpusUtf8Bytes`)
 * are throttled on the legacy 256 KiB stride so existing TUI consumers see the
 * same cadence they do today.
 */
export async function materializeFilesIntoSlotPool(
  absPaths: readonly string[],
  S: number,
  hooks?: DiskMaterializerHooks,
): Promise<BpeLinkedTokenStream> {
  if (absPaths.length === 0) {
    throw new Error('materializeFilesIntoSlotPool: at least one file path is required');
  }

  // Pre-compute total slot capacity = sum of file sizes + (numFiles - 1) separators.
  let totalBytes = 0;
  for (const p of absPaths) {
    if (p === undefined) continue;
    const st = await stat(p);
    totalBytes += st.size;
  }
  totalBytes += Math.max(0, absPaths.length - 1);
  if (totalBytes > MAX_SLOT_CAPACITY) {
    throw new Error(
      `materializeFilesIntoSlotPool: total byte length ${String(totalBytes)} exceeds MAX_SLOT_CAPACITY ${String(MAX_SLOT_CAPACITY)}`,
    );
  }
  const stream = createLinkedTokenStream(totalBytes);

  let utf8Total = BigInt(0);
  let sinceEmitUtf8 = 0;
  let nextSlot = 0;

  for (let fi = 0; fi < absPaths.length; fi++) {
    const p = absPaths[fi];
    if (p === undefined) continue;

    if (fi > 0) {
      // Synthetic inter-file separator. Counts toward slots but NOT toward
      // `onCorpusUtf8Bytes` — matches the historical "join files with `\n`"
      // semantics of the in-memory training path.
      appendLiveSlot(stream, nextSlot, S + FILE_SEPARATOR_BYTE);
      nextSlot += 1;
    }

    const rs = createReadStream(p);
    try {
      for await (const chunk of rs) {
        const buf = chunk as Buffer;
        // Tight synchronous inner loop — one slot per byte, no per-byte await.
        for (let i = 0; i < buf.length; i++) {
          appendLiveSlot(stream, nextSlot, S + buf[i]);
          nextSlot += 1;
        }
        utf8Total += BigInt(buf.length);
        sinceEmitUtf8 += buf.length;
        if (
          hooks?.onCorpusUtf8Bytes !== undefined &&
          sinceEmitUtf8 >= CORPUS_PROGRESS_STRIDE_BYTES
        ) {
          hooks.onCorpusUtf8Bytes(utf8Total);
          sinceEmitUtf8 %= CORPUS_PROGRESS_STRIDE_BYTES;
        }
        if (hooks?.yieldToHost !== undefined) {
          await hooks.yieldToHost();
        }
      }
    } catch (e) {
      rs.destroy(e as Error);
      throw e;
    }
  }

  hooks?.onCorpusUtf8Bytes?.(utf8Total);
  hooks?.onTokenStreamBytesWritten?.(BigInt(stream.liveCount * 4));
  return stream;
}
