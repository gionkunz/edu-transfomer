import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

const MANIFEST_VERSION = 1 as const;
const JOIN_BETWEEN_FILES = '\n';

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
  if (
    a.formatVersion !== b.formatVersion ||
    a.joinBetweenFiles !== b.joinBetweenFiles ||
    a.tokenizerResolved !== b.tokenizerResolved ||
    a.tokenizerMtimeMs !== b.tokenizerMtimeMs ||
    a.corpusFiles.length !== b.corpusFiles.length
  ) {
    return false;
  }
  for (let i = 0; i < a.corpusFiles.length; i++) {
    const x = a.corpusFiles[i];
    const y = b.corpusFiles[i];
    if (x === undefined || y === undefined) return false;
    if (x.path !== y.path || x.size !== y.size || x.mtimeMs !== y.mtimeMs) return false;
  }
  return true;
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

function readIdsLittleEndian(buf: Buffer, tokenCount: number): Uint32Array | null {
  if (buf.length !== tokenCount * 4) return null;
  const out = new Uint32Array(tokenCount);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let i = 0; i < tokenCount; i++) {
    out[i] = dv.getUint32(i * 4, true);
  }
  return out;
}

export async function tryReadCachedCorpusTokens(
  cacheBase: string,
  fingerprint: CorpusTokenCacheFingerprint,
): Promise<{ readonly ids: Uint32Array; readonly corpusUtf16Units: number } | null> {
  try {
    const [metaRaw, idsBuf] = await Promise.all([
      readFile(metaJsonPath(cacheBase), 'utf8'),
      readFile(idsBinPath(cacheBase)),
    ]);
    const parsed: unknown = JSON.parse(metaRaw) as unknown;
    if (!isManifest(parsed)) return null;

    const fpDisk = fingerprintFromManifest(parsed);
    if (!corpusTokenFingerprintsEqual(fingerprint, fpDisk)) return null;

    const ids = readIdsLittleEndian(idsBuf, parsed.tokenCount);
    if (ids === null) return null;

    return { ids, corpusUtf16Units: parsed.corpusUtf16Units };
  } catch {
    return null;
  }
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

  const buf = Buffer.allocUnsafe(ids.length * 4);
  for (let i = 0; i < ids.length; i++) {
    buf.writeUInt32LE(ids[i] as number, i * 4);
  }
  await writeFile(idsBinPath(cacheBase), buf);
}
