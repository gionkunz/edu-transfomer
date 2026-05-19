import { mkdir, open, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

const MANIFEST_VERSION = 1 as const;
const JOIN_BETWEEN_FILES = '\n';
/** Chunk size for `.ids` read/write — stays under Node's ~2 GiB `Buffer`/`readFile` limit. */
const IDS_IO_CHUNK_BYTES = 64 * 1024 * 1024;

export type CorpusFileSnap = {
  readonly path: string;
  readonly size: number;
  readonly mtimeMs: number;
};

/** Everything that must match to trust an on-disk **`.ids`** blob. */
export type CorpusTokenCacheFingerprint = {
  readonly formatVersion: typeof MANIFEST_VERSION;
  readonly joinBetweenFiles: typeof JOIN_BETWEEN_FILES;
  readonly tokenizerResolved: string;
  readonly tokenizerMtimeMs: number;
  readonly corpusFiles: readonly CorpusFileSnap[];
};

/** Full sidecar written next to **`.ids`** (little-endian **`uint32`** payload). */
export type CorpusTokenCacheManifest = CorpusTokenCacheFingerprint & {
  readonly tokenCount: number;
  readonly corpusUtf16Units: number;
};

function metaJsonPath(cacheBase: string): string {
  return `${cacheBase}.meta.json`;
}

function idsBinPath(cacheBase: string): string {
  return `${cacheBase}.ids`;
}

export async function buildCorpusTokenCacheFingerprint(opts: {
  readonly tokenizerPath: string;
  readonly corpusRelPaths: readonly string[];
  readonly cwd?: string;
}): Promise<CorpusTokenCacheFingerprint> {
  const cwd = opts.cwd ?? process.cwd();
  const tokenizerResolved = path.resolve(opts.tokenizerPath);
  const tokSt = await stat(tokenizerResolved);

  const corpusFiles: CorpusFileSnap[] = [];
  for (const rel of opts.corpusRelPaths) {
    const p = path.resolve(cwd, rel);
    const st = await stat(p);
    corpusFiles.push({ path: p, size: st.size, mtimeMs: st.mtimeMs });
  }

  return {
    formatVersion: MANIFEST_VERSION,
    joinBetweenFiles: JOIN_BETWEEN_FILES,
    tokenizerResolved,
    tokenizerMtimeMs: tokSt.mtimeMs,
    corpusFiles,
  };
}

export function corpusTokenFingerprintsEqual(
  a: CorpusTokenCacheFingerprint,
  b: CorpusTokenCacheFingerprint,
): boolean {
  return describeCorpusTokenFingerprintMismatch(a, b).length === 0;
}

const DEFAULT_MAX_CORPUS_FILE_DIFFS = 3;

/**
 * Human-readable reasons why **`expected`** (this run) does not match **`disk`** (manifest).
 * Empty array means the fingerprints are equivalent.
 */
export function describeCorpusTokenFingerprintMismatch(
  expected: CorpusTokenCacheFingerprint,
  disk: CorpusTokenCacheFingerprint,
  opts?: { readonly maxCorpusFileDiffs?: number },
): string[] {
  const maxCorpusFileDiffs = opts?.maxCorpusFileDiffs ?? DEFAULT_MAX_CORPUS_FILE_DIFFS;
  const reasons: string[] = [];

  if (expected.formatVersion !== disk.formatVersion) {
    reasons.push(
      `cache formatVersion ${String(disk.formatVersion)} ≠ current ${String(expected.formatVersion)}`,
    );
  }
  if (expected.joinBetweenFiles !== disk.joinBetweenFiles) {
    reasons.push('cache joinBetweenFiles differs from current build');
  }
  if (expected.tokenizerResolved !== disk.tokenizerResolved) {
    reasons.push(
      `tokenizer path: cache ${disk.tokenizerResolved} ≠ current ${expected.tokenizerResolved}`,
    );
  }
  if (expected.tokenizerMtimeMs !== disk.tokenizerMtimeMs) {
    reasons.push(
      `tokenizer mtime: cache ${String(disk.tokenizerMtimeMs)} ≠ current ${String(expected.tokenizerMtimeMs)} (file was modified)`,
    );
  }
  if (expected.corpusFiles.length !== disk.corpusFiles.length) {
    reasons.push(
      `corpus file count: cache ${String(disk.corpusFiles.length)} ≠ current ${String(expected.corpusFiles.length)} (--data glob may match different files)`,
    );
    return reasons;
  }

  let corpusDiffs = 0;
  for (let i = 0; i < expected.corpusFiles.length; i++) {
    const x = expected.corpusFiles[i];
    const y = disk.corpusFiles[i];
    if (x === undefined || y === undefined) {
      reasons.push(`corpus file #${String(i)} missing from fingerprint`);
      corpusDiffs++;
      continue;
    }
    const label = path.basename(x.path);
    if (x.path !== y.path) {
      if (corpusDiffs < maxCorpusFileDiffs) {
        reasons.push(
          `${label}: path cache ${y.path} ≠ current ${x.path} (cwd may differ — use absolute --token-cache or run from the same directory)`,
        );
      }
      corpusDiffs++;
      continue;
    }
    if (x.size !== y.size) {
      if (corpusDiffs < maxCorpusFileDiffs) {
        reasons.push(`${label}: size ${String(y.size)} in cache ≠ current ${String(x.size)}`);
      }
      corpusDiffs++;
      continue;
    }
    if (x.mtimeMs !== y.mtimeMs) {
      if (corpusDiffs < maxCorpusFileDiffs) {
        reasons.push(
          `${label}: mtime ${String(y.mtimeMs)} in cache ≠ current ${String(x.mtimeMs)} (file was modified)`,
        );
      }
      corpusDiffs++;
    }
  }
  if (corpusDiffs > maxCorpusFileDiffs) {
    reasons.push(`… and ${String(corpusDiffs - maxCorpusFileDiffs)} more corpus file mismatch(es)`);
  }

  return reasons;
}

function readErrnoMessage(e: unknown, artifactPath: string): string {
  if (
    e !== null &&
    typeof e === 'object' &&
    'code' in e &&
    (e as NodeJS.ErrnoException).code === 'ENOENT'
  ) {
    return `${artifactPath} not found (check cwd or use an absolute --token-cache path)`;
  }
  const msg = e instanceof Error ? e.message : String(e);
  return `failed to read ${artifactPath}: ${msg}`;
}

function fingerprintFromManifest(m: CorpusTokenCacheManifest): CorpusTokenCacheFingerprint {
  return {
    formatVersion: m.formatVersion,
    joinBetweenFiles: m.joinBetweenFiles,
    tokenizerResolved: m.tokenizerResolved,
    tokenizerMtimeMs: m.tokenizerMtimeMs,
    corpusFiles: m.corpusFiles,
  };
}

function isManifest(raw: unknown): raw is CorpusTokenCacheManifest {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) return false;
  const o = raw as Record<string, unknown>;
  const files = o.corpusFiles;
  if (
    typeof o.tokenizerResolved !== 'string' ||
    typeof o.tokenizerMtimeMs !== 'number' ||
    typeof o.tokenCount !== 'number' ||
    typeof o.corpusUtf16Units !== 'number' ||
    o.formatVersion !== MANIFEST_VERSION ||
    o.joinBetweenFiles !== JOIN_BETWEEN_FILES ||
    !Array.isArray(files)
  ) {
    return false;
  }
  for (const ent of files) {
    if (ent === null || typeof ent !== 'object' || Array.isArray(ent)) return false;
    const e = ent as Record<string, unknown>;
    if (typeof e.path !== 'string' || typeof e.size !== 'number' || typeof e.mtimeMs !== 'number') {
      return false;
    }
  }
  return true;
}

function hostStoresUint32LittleEndian(): boolean {
  const probe = new Uint8Array(new Uint32Array([0x04030201]).buffer);
  return probe[0] === 1;
}

async function readIdsPayloadLittleEndian(
  idsPath: string,
  tokenCount: number,
): Promise<Uint32Array | null> {
  const expectedBytes = tokenCount * 4;
  let fileSize: number;
  try {
    fileSize = (await stat(idsPath)).size;
  } catch {
    return null;
  }
  if (fileSize !== expectedBytes) {
    return null;
  }

  if (expectedBytes >= 512 * 1024 * 1024) {
    const gib = expectedBytes / (1024 * 1024 * 1024);
    console.error(
      `train: loading token cache payload (${gib.toFixed(1)} GiB, ${String(tokenCount)} ids)…`,
    );
  }

  if (!hostStoresUint32LittleEndian()) {
    return readIdsPayloadLittleEndianPortable(idsPath, tokenCount, expectedBytes);
  }

  let ids: Uint32Array;
  try {
    ids = new Uint32Array(tokenCount);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(
      `corpus token cache: failed to allocate ${String(tokenCount)} ids (${String(expectedBytes)} bytes): ${msg}`,
    );
  }

  const bytes = new Uint8Array(ids.buffer, ids.byteOffset, ids.byteLength);
  const fh = await open(idsPath, 'r');
  try {
    let offset = 0;
    while (offset < expectedBytes) {
      const want = Math.min(IDS_IO_CHUNK_BYTES, expectedBytes - offset);
      const { bytesRead } = await fh.read(bytes, offset, want, offset);
      if (bytesRead === 0) {
        return null;
      }
      offset += bytesRead;
    }
  } finally {
    await fh.close();
  }
  return ids;
}

/** Big-endian hosts: read LE uint32 via `DataView` in chunked passes. */
async function readIdsPayloadLittleEndianPortable(
  idsPath: string,
  tokenCount: number,
  expectedBytes: number,
): Promise<Uint32Array | null> {
  const ids = new Uint32Array(tokenCount);
  const chunkBytes = IDS_IO_CHUNK_BYTES - (IDS_IO_CHUNK_BYTES % 4);
  const chunk = Buffer.allocUnsafe(chunkBytes);
  const fh = await open(idsPath, 'r');
  try {
    let fileOffset = 0;
    let tokenIndex = 0;
    while (fileOffset < expectedBytes) {
      const want = Math.min(chunkBytes, expectedBytes - fileOffset);
      const { bytesRead } = await fh.read(chunk, 0, want, fileOffset);
      if (bytesRead === 0) {
        return null;
      }
      const dv = new DataView(chunk.buffer, chunk.byteOffset, bytesRead);
      for (let i = 0; i + 4 <= bytesRead; i += 4) {
        ids[tokenIndex] = dv.getUint32(i, true);
        tokenIndex++;
      }
      fileOffset += bytesRead;
    }
    if (tokenIndex !== tokenCount) {
      return null;
    }
  } finally {
    await fh.close();
  }
  return ids;
}

async function writeIdsPayloadLittleEndian(idsPath: string, ids: Uint32Array): Promise<void> {
  const byteLength = ids.byteLength;
  if (!hostStoresUint32LittleEndian()) {
    await writeIdsPayloadLittleEndianPortable(idsPath, ids);
    return;
  }

  const bytes = new Uint8Array(ids.buffer, ids.byteOffset, byteLength);
  const fh = await open(idsPath, 'w');
  try {
    let offset = 0;
    while (offset < byteLength) {
      const want = Math.min(IDS_IO_CHUNK_BYTES, byteLength - offset);
      const { bytesWritten } = await fh.write(bytes, offset, want, offset);
      if (bytesWritten === 0) {
        throw new Error(`corpus token cache: write stalled at byte offset ${String(offset)}`);
      }
      offset += bytesWritten;
    }
  } finally {
    await fh.close();
  }
}

async function writeIdsPayloadLittleEndianPortable(
  idsPath: string,
  ids: Uint32Array,
): Promise<void> {
  const chunkTokens = Math.floor(IDS_IO_CHUNK_BYTES / 4);
  const chunk = Buffer.allocUnsafe(chunkTokens * 4);
  const fh = await open(idsPath, 'w');
  try {
    let tokenOffset = 0;
    while (tokenOffset < ids.length) {
      const n = Math.min(chunkTokens, ids.length - tokenOffset);
      for (let i = 0; i < n; i++) {
        chunk.writeUInt32LE(ids[tokenOffset + i] ?? 0, i * 4);
      }
      const bytes = n * 4;
      const fileOffset = tokenOffset * 4;
      const { bytesWritten } = await fh.write(chunk, 0, bytes, fileOffset);
      if (bytesWritten !== bytes) {
        throw new Error(
          `corpus token cache: short write at token offset ${String(tokenOffset)} (wrote ${String(bytesWritten)}/${String(bytes)} bytes)`,
        );
      }
      tokenOffset += n;
    }
  } finally {
    await fh.close();
  }
}

export type CorpusTokenCacheReadResult =
  | {
      readonly status: 'hit';
      readonly ids: Uint32Array;
      readonly corpusUtf16Units: number;
    }
  | {
      readonly status: 'miss';
      readonly reasons: readonly string[];
    };

/** Resolve cache artifact paths for stderr diagnostics. */
export function corpusTokenCacheArtifactPaths(cacheBase: string): {
  readonly metaJson: string;
  readonly idsBin: string;
} {
  return {
    metaJson: metaJsonPath(cacheBase),
    idsBin: idsBinPath(cacheBase),
  };
}

export async function readCorpusTokenCache(
  cacheBase: string,
  fingerprint: CorpusTokenCacheFingerprint,
): Promise<CorpusTokenCacheReadResult> {
  const paths = corpusTokenCacheArtifactPaths(cacheBase);

  let metaRaw: string;
  try {
    metaRaw = await readFile(paths.metaJson, 'utf8');
  } catch (e) {
    return { status: 'miss', reasons: [readErrnoMessage(e, paths.metaJson)] };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(metaRaw) as unknown;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { status: 'miss', reasons: [`${paths.metaJson} is not valid JSON: ${msg}`] };
  }

  if (!isManifest(parsed)) {
    return {
      status: 'miss',
      reasons: [`${paths.metaJson} is missing required fields or uses an unsupported schema`],
    };
  }

  const fpDisk = fingerprintFromManifest(parsed);
  const fpMismatch = describeCorpusTokenFingerprintMismatch(fingerprint, fpDisk);
  if (fpMismatch.length > 0) {
    return { status: 'miss', reasons: fpMismatch };
  }

  let idsFileSize: number;
  try {
    idsFileSize = (await stat(paths.idsBin)).size;
  } catch (e) {
    return { status: 'miss', reasons: [readErrnoMessage(e, paths.idsBin)] };
  }

  const expectedBytes = parsed.tokenCount * 4;
  if (idsFileSize !== expectedBytes) {
    return {
      status: 'miss',
      reasons: [
        `${paths.idsBin} size ${String(idsFileSize)} bytes ≠ expected ${String(expectedBytes)} (${String(parsed.tokenCount)} tokens × 4)`,
      ],
    };
  }

  let ids: Uint32Array | null;
  try {
    ids = await readIdsPayloadLittleEndian(paths.idsBin, parsed.tokenCount);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { status: 'miss', reasons: [`failed to read ${paths.idsBin}: ${msg}`] };
  }
  if (ids === null) {
    return {
      status: 'miss',
      reasons: [
        `${paths.idsBin} could not be read as ${String(parsed.tokenCount)} little-endian u32 values`,
      ],
    };
  }

  return { ids, corpusUtf16Units: parsed.corpusUtf16Units, status: 'hit' };
}

export async function writeCachedCorpusTokens(
  cacheBase: string,
  fingerprint: CorpusTokenCacheFingerprint,
  ids: Uint32Array,
  corpusUtf16Units: number,
): Promise<void> {
  await mkdir(path.dirname(path.resolve(cacheBase)), { recursive: true });

  const manifest: CorpusTokenCacheManifest = {
    ...fingerprint,
    tokenCount: ids.length,
    corpusUtf16Units,
  };
  await writeFile(metaJsonPath(cacheBase), `${JSON.stringify(manifest, null, 2)}\n`);

  await writeIdsPayloadLittleEndian(idsBinPath(cacheBase), ids);
}
