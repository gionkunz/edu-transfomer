import { createReadStream, createWriteStream, type WriteStream } from 'node:fs';
import { open, unlink } from 'node:fs/promises';
import type { Readable } from 'node:stream';
import { finished } from 'node:stream/promises';

import { bpePairKey } from './bpe-merge-stats';

const TOKEN_READ_CHUNK_BYTES = Math.max(65536, 256 * 1024);
/** Tokens per flushed write (**256 KiB**) — balances fewer syscalls vs back-pressure granularity. */
const WRITE_BATCH_TOKENS = 65536;

/** Emit corpus-read progress at most every this many UTF-8 bytes (~256 KiB). */
const CORPUS_PROGRESS_STRIDE_BYTES = 256 * 1024;

/** Emit token-file write progress (~4 MiB **`WriteStream.bytesWritten`** steps). */
const TOKEN_FILE_WRITE_STRIDE_BYTES = 4 * 1024 * 1024;

async function unlinkQuiet(p: string): Promise<void> {
  try {
    await unlink(p);
  } catch {
    /* best-effort */
  }
}

/** Flushes one buffered chunk via **`writable.write(..., cb)`** (back-pressure safe). */
async function writeWholeBufferedChunk(ws: WriteStream, data: Uint8Array): Promise<void> {
  if (data.byteLength === 0) return;
  const chunk = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  await new Promise<void>((resolve, reject) => {
    ws.write(chunk, (err: Error | null | undefined) => {
      if (err !== null && err !== undefined) reject(err);
      else resolve();
    });
  });
}

class U32LeWriteBatcher {
  private readonly slab: Buffer = Buffer.allocUnsafe(WRITE_BATCH_TOKENS * 4);
  private count = 0;

  constructor(private readonly onAfterWrite?: (ws: WriteStream) => void) {}

  async push(ws: WriteStream, tok: number): Promise<void> {
    this.slab.writeUInt32LE(tok >>> 0, this.count << 2);
    this.count += 1;
    if (this.count >= WRITE_BATCH_TOKENS) {
      await writeWholeBufferedChunk(
        ws,
        new Uint8Array(this.slab.buffer, this.slab.byteOffset, this.count << 2),
      );
      this.count = 0;
      this.onAfterWrite?.(ws);
    }
  }

  /**
   * Synchronous bulk-push: read up to `(WRITE_BATCH_TOKENS - count)` bytes from
   * `srcBuf` starting at `srcOffset`, encoding each as `byteTokenOffset + b`
   * into the slab. Returns the number of bytes consumed from `srcBuf`. Caller
   * must check {@link isFull} and call {@link flushFullBatch} between rounds.
   */
  pushUntilFull(byteTokenOffset: number, srcBuf: Buffer, srcOffset: number): number {
    const room = WRITE_BATCH_TOKENS - this.count;
    const consumeable = Math.min(room, srcBuf.length - srcOffset);
    const slab = this.slab;
    let pos = this.count << 2;
    for (let k = 0; k < consumeable; k++) {
      slab.writeUInt32LE((byteTokenOffset + srcBuf[srcOffset + k]!) >>> 0, pos);
      pos += 4;
    }
    this.count += consumeable;
    return consumeable;
  }

  isFull(): boolean {
    return this.count >= WRITE_BATCH_TOKENS;
  }

  async flushFullBatch(ws: WriteStream): Promise<void> {
    if (this.count === 0) return;
    await writeWholeBufferedChunk(
      ws,
      new Uint8Array(this.slab.buffer, this.slab.byteOffset, this.count << 2),
    );
    this.count = 0;
    this.onAfterWrite?.(ws);
  }

  async flush(ws: WriteStream): Promise<void> {
    if (this.count === 0) return;
    await writeWholeBufferedChunk(
      ws,
      new Uint8Array(this.slab.buffer, this.slab.byteOffset, this.count << 2),
    );
    this.count = 0;
    this.onAfterWrite?.(ws);
  }
}

/**
 * Drain one `createReadStream` chunk into the `u32` LE batcher. Walks the chunk
 * synchronously (no per-byte `await`); awaits at most once at the end of the
 * chunk to flush whole batches that crossed the `WRITE_BATCH_TOKENS` boundary.
 * Eliminates the microtask storm that previously throttled streaming throughput
 * to ~1.5 M tokens/sec.
 */
async function drainChunkIntoBatch(
  ws: WriteStream,
  batch: U32LeWriteBatcher,
  byteTokenOffset: number,
  buf: Buffer,
): Promise<void> {
  let i = 0;
  while (i < buf.length) {
    const advance = batch.pushUntilFull(byteTokenOffset, buf, i);
    i += advance;
    if (batch.isFull()) {
      await batch.flushFullBatch(ws);
    }
  }
}

async function pipelineReadStreamToTokens(
  rs: Readable,
  ws: WriteStream,
  batch: U32LeWriteBatcher,
  byteTokenOffset: number,
  onUtf8Chunk?: (chunkByteLength: number) => void,
): Promise<void> {
  try {
    for await (const chunk of rs) {
      const buf = chunk as Buffer;
      onUtf8Chunk?.(buf.length);
      await drainChunkIntoBatch(ws, batch, byteTokenOffset, buf);
    }
  } catch (e) {
    rs.destroy(e as Error);
    throw e;
  }
}

/**
 * Join UTF-8 files as contiguous **`uint32` LE token stream**: each **`b`** → **`byteTokenOffset + b`**.
 * Mirrors **`Tokenizer.train`** initial mapping; inserts UTF-8 **`\n`** (byte **`10`**) **between files**.
 *
 * Inner loop walks each delivered `Buffer` chunk synchronously through
 * {@link U32LeWriteBatcher.pushUntilFull} and `await`s only when a full
 * `WRITE_BATCH_TOKENS` block needs flushing — no per-byte microtask anymore.
 *
 * NOTE: After the incremental BPE training change, this writer is no longer on
 * the hot path of `BPETokenizer.trainFromFiles` (which now materializes directly
 * into an in-memory linked-list slot pool). Kept exported for ad-hoc tooling
 * and for symmetry with the on-disk frequency / rewrite helpers below.
 */
export async function writeJoinedCorpusAsTokenIdsToFile(opts: {
  readonly filePaths: readonly string[];
  readonly outPath: string;
  readonly byteTokenOffset: number;
  /** Throttled (**~256 KiB** steps) while reading corpus UTF-8 (**excludes** synthetic **`\\n`** between files). Final call is exact total. */
  readonly onCorpusUtf8Bytes?: (cumulativeUtf8Bytes: bigint) => void;
  /**
   * Bytes accepted into the **`u32` LE** output file (see **`WriteStream.bytesWritten`**).
   * Throttled (~**`TOKEN_FILE_WRITE_STRIDE_BYTES`**) so hosts can show progress while the OS drains a huge write buffer.
   */
  readonly onTokenStreamBytesWritten?: (cumulativeTokenFileBytes: bigint) => void;
}): Promise<void> {
  const { filePaths, outPath, byteTokenOffset, onCorpusUtf8Bytes, onTokenStreamBytesWritten } =
    opts;
  let utf8Total = BigInt(0);
  let sinceEmitUtf8 = 0;

  let lastWriteEmit = 0;
  const onAfterWrite =
    onTokenStreamBytesWritten === undefined
      ? undefined
      : (w: WriteStream) => {
          const bw = w.bytesWritten;
          if (bw - lastWriteEmit >= TOKEN_FILE_WRITE_STRIDE_BYTES) {
            lastWriteEmit = bw;
            onTokenStreamBytesWritten(BigInt(bw));
          }
        };

  const ws = createWriteStream(outPath, { flags: 'wx', highWaterMark: 512 * 1024 });
  const batch = new U32LeWriteBatcher(onAfterWrite);
  try {
    for (let fi = 0; fi < filePaths.length; fi++) {
      const p = filePaths[fi];
      if (p === undefined) continue;

      if (fi > 0) {
        await batch.push(ws, byteTokenOffset + 10);
      }

      const rs = createReadStream(p);
      await pipelineReadStreamToTokens(rs, ws, batch, byteTokenOffset, (n) => {
        utf8Total += BigInt(n);
        sinceEmitUtf8 += n;
        if (onCorpusUtf8Bytes !== undefined && sinceEmitUtf8 >= CORPUS_PROGRESS_STRIDE_BYTES) {
          onCorpusUtf8Bytes(utf8Total);
          sinceEmitUtf8 %= CORPUS_PROGRESS_STRIDE_BYTES;
        }
      });
    }

    onCorpusUtf8Bytes?.(utf8Total);
    await batch.flush(ws);
    onAfterWrite?.(ws);
    ws.end();
    await finished(ws);
    onTokenStreamBytesWritten?.(BigInt(ws.bytesWritten));
  } catch (e) {
    ws.destroy();
    throw e;
  }
}

/**
 * Adjacent overlapping pair counts for a **`uint32` LE token file** (**4-byte aligned**, little-endian).
 *
 * @deprecated O(N) per call and goes through the legacy `Map<string, number>`
 * keyed by {@link bpePairKey}. The incremental training pipeline in
 * `packages/tokenizer/src/bpe-train-incremental/` maintains pair counts via a
 * typed-array open-addressing hashmap and only ever scans the corpus once;
 * this helper is retained for ad-hoc disk-stream forensics.
 */
export async function countAdjacentPairsFromU32LeTokenFile(
  absPath: string,
): Promise<Map<string, number>> {
  const freq = new Map<string, number>();
  const fh = await open(absPath, 'r');
  try {
    const stat = await fh.stat();
    const size = stat.size;
    if (size % 4 !== 0) {
      throw new Error(`bpe-disk: token stream size ${String(size)} is not multiple of 4`);
    }
    if (size < 8) return freq;

    let prev: number | undefined;
    let fileOffset = 0;

    while (fileOffset < size) {
      const toRead = Math.min(TOKEN_READ_CHUNK_BYTES, size - fileOffset);
      const slab = Buffer.allocUnsafe(toRead);
      const got = await fh.read(slab, 0, toRead, fileOffset);
      if (got.bytesRead === 0) break;

      const nBytes = got.bytesRead;
      if (nBytes % 4 !== 0) {
        throw new Error('bpe-disk: read chunk not multiple-of-4 — corrupted stream');
      }
      fileOffset += nBytes;

      const u32Count = nBytes >>> 2;
      for (let i = 0; i < u32Count; i++) {
        const cur = slab.readUInt32LE(i << 2);
        if (prev !== undefined) {
          const pk = bpePairKey(prev, cur);
          freq.set(pk, (freq.get(pk) ?? 0) + 1);
        }
        prev = cur;
      }
    }
  } finally {
    await fh.close();
  }
  return freq;
}

/**
 * Greedy **`L+R → newId`** rewrite (**same semantics** as in-memory **`rewriteGreedy`**): streaming read,
 * streamed **`uint32` LE** to **`outPath`** (**overwrites** **`outPath`** if present).
 *
 * @deprecated O(N) per call and was the single largest contributor to the
 * `O(N · M)` blow-up that motivated the incremental BPE training pipeline.
 * The production path under `packages/tokenizer/src/bpe-train-incremental/`
 * applies merges in `O(K_m)` against an in-memory linked-list slot pool;
 * this helper is retained for ad-hoc disk-stream forensics and parity tests.
 */
export async function rewriteGreedyU32LeTokenFile(opts: {
  readonly inPath: string;
  readonly outPath: string;
  readonly L: number;
  readonly R: number;
  readonly newId: number;
}): Promise<void> {
  const { inPath, outPath, L, R, newId } = opts;

  await unlinkQuiet(outPath);
  const inFh = await open(inPath, 'r');

  try {
    const stat = await inFh.stat();
    const byteSize = stat.size;
    if (byteSize % 4 !== 0) {
      throw new Error(`bpe-disk: token stream size ${String(byteSize)} is not multiple of 4`);
    }

    const ws = createWriteStream(outPath, { flags: 'wx', highWaterMark: 512 * 1024 });
    const batch = new U32LeWriteBatcher();
    let readOffset = 0;
    let slack = Buffer.alloc(0);

    try {
      const refill = async (): Promise<void> => {
        while (slack.length < 4 && readOffset < byteSize) {
          const chunkSize = Math.min(TOKEN_READ_CHUNK_BYTES, byteSize - readOffset);
          const slab = Buffer.allocUnsafe(chunkSize);
          const got = await inFh.read(slab, 0, chunkSize, readOffset);
          if (got.bytesRead === 0) {
            throw new Error('bpe-disk: unexpected EOF while reading token stream');
          }
          readOffset += got.bytesRead;
          slack =
            slack.length === 0
              ? slab.subarray(0, got.bytesRead)
              : Buffer.concat([slack, slab.subarray(0, got.bytesRead)]);
        }
      };

      const nextU32 = async (): Promise<number | undefined> => {
        await refill();
        if (slack.length < 4) {
          if (slack.length !== 0) {
            throw new Error('bpe-disk: stray trailing bytes in rewrite input');
          }
          return undefined;
        }
        const v = slack.readUInt32LE(0);
        slack = slack.subarray(4);
        return v;
      };

      let a = await nextU32();
      while (a !== undefined) {
        const b = await nextU32();
        if (b === undefined) {
          await batch.push(ws, a);
          break;
        }
        if (a === L && b === R) {
          await batch.push(ws, newId);
          a = await nextU32();
        } else {
          await batch.push(ws, a);
          a = b;
        }
      }
      await batch.flush(ws);
      ws.end();
      await finished(ws);
    } catch (e) {
      ws.destroy();
      throw e;
    }
  } finally {
    await inFh.close();
  }
}
