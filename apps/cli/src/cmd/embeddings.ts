import { readFile } from 'node:fs/promises';

import {
  assertEmbeddingMatchesTokenizer,
  classifyCosine,
  collectAtomNodes,
  cosineSimilarity,
  cosineStatsAgainstMatrix,
  describeSimilarityBand,
  EmbeddingDimensionError,
  evaluateEmbeddingExpression,
  lookupEmbedding,
  makeEmbeddingMatrix,
  parseEmbeddingExpression,
  topKByCosine,
  type CosineStats,
  type EmbeddingMatrix,
  type ResolveAtom,
  type SimilarityBand,
  type TopKEntry,
} from '@llm-transformer-diy/inference';
import { readPackedCheckpointFiles, type PackedCheckpointSidecar } from '@llm-transformer-diy/io';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';

import { resolvePackedCheckpointPaths } from '../paths';

const TOKEN_EMBED_TENSOR_NAME = 'token_embed';

/**
 * Words almost always appear with a leading space in GPT-2-style BPE corpora
 * (the pretokenizer keeps the leading space attached, so " king" and "king"
 * are different tokens — and " king" is the variant seen in real text).
 * This helper applies that prefix when the user hasn't explicitly provided one
 * and the input isn't already a `@file` vector spec.
 */
function applyAutoSpace(text: string, autoSpace: boolean): string {
  if (!autoSpace) return text;
  if (text.length === 0) return text;
  if (text.startsWith('@')) return text;
  if (/^\s/.test(text)) return text;
  return ` ${text}`;
}

/**
 * Loaded artifacts shared by every `llm embeddings *` subcommand: the tokenizer,
 * the input token-embedding matrix (CPU, **`Float32Array`**), and a tokenize-and-pool
 * helper that turns an atom string (word or quoted phrase) into a single vector.
 */
type EmbeddingsContext = {
  readonly tokenizer: BPETokenizer;
  readonly matrix: EmbeddingMatrix;
  readonly tokenize: (text: string) => { ids: number[]; surfaces: string[] };
};

function dModelFromConfig(config: unknown): number | undefined {
  if (config === null || typeof config !== 'object') return undefined;
  const v = (config as { dModel?: unknown }).dModel;
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function extractEmbeddingMatrix(
  tensors: Record<string, Float32Array>,
  sidecar: PackedCheckpointSidecar,
): EmbeddingMatrix {
  const data = tensors[TOKEN_EMBED_TENSOR_NAME];
  if (data === undefined) {
    throw new Error(
      `embeddings: checkpoint is missing tensor "${TOKEN_EMBED_TENSOR_NAME}" (have: ${Object.keys(tensors).sort().join(', ')})`,
    );
  }
  const entry = sidecar.tensors[TOKEN_EMBED_TENSOR_NAME];
  if (!entry) {
    throw new Error(`embeddings: manifest missing entry for "${TOKEN_EMBED_TENSOR_NAME}"`);
  }
  if (entry.shape.length !== 2) {
    throw new Error(
      `embeddings: expected 2-D token-embedding tensor, got shape [${entry.shape.join(', ')}]`,
    );
  }
  const [vocab, dModel] = entry.shape as [number, number];
  const dExpected = dModelFromConfig(sidecar.config);
  if (dExpected !== undefined && dExpected !== dModel) {
    throw new Error(
      `embeddings: config dModel=${String(dExpected)} ≠ token_embed shape [${String(vocab)}, ${String(dModel)}]`,
    );
  }
  return makeEmbeddingMatrix(data, vocab, dModel);
}

async function loadEmbeddingsContext(opts: {
  readonly modelArg: string;
  readonly tokenizerPath: string;
}): Promise<EmbeddingsContext> {
  const { binPath, jsonPath } = resolvePackedCheckpointPaths(opts.modelArg);
  const { sidecar, tensors } = await readPackedCheckpointFiles(binPath, jsonPath);
  const matrix = extractEmbeddingMatrix(tensors, sidecar);

  const tokenizer = new BPETokenizer();
  await tokenizer.load(opts.tokenizerPath);
  assertEmbeddingMatchesTokenizer(matrix, tokenizer);

  const tokenize = (text: string): { ids: number[]; surfaces: string[] } => {
    const ids = tokenizer.encode(text);
    if (ids.length === 0) {
      throw new Error(`embeddings: input ${JSON.stringify(text)} tokenized to 0 ids`);
    }
    const surfaces = ids.map((id) => tokenizer.decode([id], { utf8: 'replace' }));
    return { ids, surfaces };
  };

  return { tokenizer, matrix, tokenize };
}

/**
 * Loads `@<path>` JSON spec containing **`{ "vector": [..] }`**, otherwise treats `spec`
 * as a phrase to encode + mean-pool. Returns the vector and an input-trace label.
 */
async function resolveVectorSpec(
  ctx: EmbeddingsContext,
  spec: string,
  autoSpace: boolean,
): Promise<{
  vector: Float32Array;
  label: string;
  /** Token ids consumed (empty when the vector came from a file). */
  tokenIds: number[];
}> {
  if (spec.startsWith('@')) {
    const filePath = spec.slice(1);
    if (filePath === '') {
      throw new Error('embeddings: empty file path after `@`');
    }
    const buf = await readFile(filePath, 'utf8');
    const parsed: unknown = JSON.parse(buf);
    if (parsed === null || typeof parsed !== 'object' || !('vector' in parsed)) {
      throw new Error(`embeddings: ${filePath} must contain {"vector": [..]}`);
    }
    const v = (parsed as { vector: unknown }).vector;
    if (!Array.isArray(v) || v.some((x) => typeof x !== 'number')) {
      throw new Error(`embeddings: ${filePath} "vector" must be an array of numbers`);
    }
    if (v.length !== ctx.matrix.dModel) {
      throw new EmbeddingDimensionError(ctx.matrix.dModel, v.length, `vector from ${filePath}`);
    }
    return { vector: Float32Array.from(v as number[]), label: `@${filePath}`, tokenIds: [] };
  }
  const encoded = applyAutoSpace(spec, autoSpace);
  const { ids, surfaces } = ctx.tokenize(encoded);
  const encodedNote = encoded === spec ? '' : ` → encoded as ${JSON.stringify(encoded)}`;
  const vec = lookupEmbedding(ctx.matrix, ids, { pool: 'mean' }) as Float32Array;
  const trace = `${JSON.stringify(spec)}${encodedNote} → ${String(ids.length)} token(s): ${ids
    .map((id, i) => `${String(id)}=${JSON.stringify(surfaces[i] ?? '')}`)
    .join(', ')}${ids.length > 1 ? ' [mean-pooled]' : ''}`;
  return { vector: vec, label: trace, tokenIds: ids };
}

function decodeId(tokenizer: BPETokenizer, id: number): string {
  return tokenizer.decode([id], { utf8: 'replace' });
}

function formatVectorOneLine(v: Float32Array): string {
  const parts: string[] = [];
  for (let i = 0; i < v.length; i++) parts.push(formatFloat(v[i] as number));
  return `[${parts.join(', ')}]`;
}

function formatFloat(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  // 6 sig digits (≥ 4 required by spec) without trailing zeros.
  return Number.parseFloat(x.toPrecision(6)).toString();
}

type LabeledTopKEntry = TopKEntry & {
  readonly z: number;
  readonly band: SimilarityBand;
};

function labelTopK(entries: ReadonlyArray<TopKEntry>, stats: CosineStats): LabeledTopKEntry[] {
  return entries.map((e) => {
    const c = classifyCosine(e.cosine, stats);
    return { ...e, z: c.z, band: c.band };
  });
}

function formatZ(z: number): string {
  if (z === Number.POSITIVE_INFINITY) return '+∞';
  if (z === Number.NEGATIVE_INFINITY) return '-∞';
  if (!Number.isFinite(z)) return 'NaN';
  const abs = Math.abs(z);
  return (z < 0 ? '-' : '') + abs.toFixed(1);
}

/** JSON-safe `z`: maps non-finite values to a string sentinel so consumers can round-trip them. */
function jsonSafeZ(z: number): number | string {
  if (z === Number.POSITIVE_INFINITY) return '+inf';
  if (z === Number.NEGATIVE_INFINITY) return '-inf';
  if (!Number.isFinite(z)) return 'NaN';
  return z;
}

function formatTopKTable(
  entries: ReadonlyArray<LabeledTopKEntry>,
  tokenizer: BPETokenizer,
): string {
  const rows = entries.map((e, i) => ({
    rank: String(i + 1),
    token: JSON.stringify(decodeId(tokenizer, e.index)),
    id: String(e.index),
    cosine: formatFloat(e.cosine),
    z: formatZ(e.z),
    band: describeSimilarityBand(e.band),
  }));
  const headers = {
    rank: 'rank',
    token: 'token',
    id: 'id',
    cosine: 'cosine',
    z: 'z',
    band: 'band',
  };
  const widths = {
    rank: Math.max(headers.rank.length, ...rows.map((r) => r.rank.length)),
    token: Math.max(headers.token.length, ...rows.map((r) => r.token.length)),
    id: Math.max(headers.id.length, ...rows.map((r) => r.id.length)),
    cosine: Math.max(headers.cosine.length, ...rows.map((r) => r.cosine.length)),
    z: Math.max(headers.z.length, ...rows.map((r) => r.z.length)),
    band: Math.max(headers.band.length, ...rows.map((r) => r.band.length)),
  };
  const fmt = (r: typeof headers): string =>
    `${r.rank.padStart(widths.rank)}  ${r.token.padEnd(widths.token)}  ${r.id.padStart(widths.id)}  ${r.cosine.padStart(widths.cosine)}  ${r.z.padStart(widths.z)}  ${r.band.padEnd(widths.band)}`;
  const lines: string[] = [fmt(headers)];
  lines.push(
    `${'-'.repeat(widths.rank)}  ${'-'.repeat(widths.token)}  ${'-'.repeat(widths.id)}  ${'-'.repeat(widths.cosine)}  ${'-'.repeat(widths.z)}  ${'-'.repeat(widths.band)}`,
  );
  for (const r of rows) lines.push(fmt(r));
  return lines.join('\n');
}

type EncodeOpts = {
  readonly modelArg: string;
  readonly tokenizerPath: string;
  readonly text: string;
  readonly perToken: boolean;
  readonly json: boolean;
  readonly autoSpace: boolean;
};

export async function runEmbeddingsEncode(opts: EncodeOpts): Promise<void> {
  const ctx = await loadEmbeddingsContext(opts);
  const text = applyAutoSpace(opts.text, opts.autoSpace);
  const { ids, surfaces } = ctx.tokenize(text);

  if (opts.perToken) {
    const rows = lookupEmbedding(ctx.matrix, ids, { pool: 'none' }) as Float32Array[];
    if (opts.json) {
      console.log(
        JSON.stringify({
          input: opts.text,
          encoded: text,
          tokens: ids.map((id, i) => ({ id, surface: surfaces[i] ?? '' })),
          vectors: rows.map((r) => Array.from(r)),
        }),
      );
      return;
    }
    console.log(
      `input ${JSON.stringify(opts.text)} → encoded as ${JSON.stringify(text)} → ${String(ids.length)} token(s): ${ids
        .map((id, i) => `${String(id)}=${JSON.stringify(surfaces[i] ?? '')}`)
        .join(', ')}`,
    );
    for (let i = 0; i < rows.length; i++) {
      console.log(
        `  [${String(i)}] id=${String(ids[i])}: ${formatVectorOneLine(rows[i] as Float32Array)}`,
      );
    }
    return;
  }

  const vec = lookupEmbedding(ctx.matrix, ids, { pool: 'mean' }) as Float32Array;
  if (opts.json) {
    console.log(
      JSON.stringify({
        input: opts.text,
        encoded: text,
        tokens: ids.map((id, i) => ({ id, surface: surfaces[i] ?? '' })),
        vector: Array.from(vec),
      }),
    );
    return;
  }
  console.log(
    `input ${JSON.stringify(opts.text)} → encoded as ${JSON.stringify(text)} → ${String(ids.length)} token(s): ${ids
      .map((id, i) => `${String(id)}=${JSON.stringify(surfaces[i] ?? '')}`)
      .join(', ')}${ids.length > 1 ? ' [mean-pooled]' : ''}`,
  );
  console.log(formatVectorOneLine(vec));
}

type DecodeOpts = {
  readonly modelArg: string;
  readonly tokenizerPath: string;
  readonly vectorSpec: string;
  readonly topK: number;
  readonly json: boolean;
  readonly autoSpace: boolean;
};

export async function runEmbeddingsDecode(opts: DecodeOpts): Promise<void> {
  const ctx = await loadEmbeddingsContext(opts);
  const resolved = await resolveVectorSpec(ctx, opts.vectorSpec, opts.autoSpace);
  const exclude = new Set<number>(resolved.tokenIds);
  const top = topKByCosine(resolved.vector, ctx.matrix, opts.topK, { exclude });
  const stats = cosineStatsAgainstMatrix(resolved.vector, ctx.matrix, { exclude });
  const labeled = labelTopK(top, stats);
  emitTopK({
    inputs: [resolved.label],
    topK: labeled,
    stats,
    json: opts.json,
    tokenizer: ctx.tokenizer,
  });
}

type SimilarityOpts = {
  readonly modelArg: string;
  readonly tokenizerPath: string;
  readonly a: string;
  readonly b: string;
  readonly json: boolean;
  readonly autoSpace: boolean;
};

export async function runEmbeddingsSimilarity(opts: SimilarityOpts): Promise<void> {
  const ctx = await loadEmbeddingsContext(opts);
  const a = await resolveVectorSpec(ctx, opts.a, opts.autoSpace);
  const b = await resolveVectorSpec(ctx, opts.b, opts.autoSpace);
  const cos = cosineSimilarity(a.vector, b.vector);
  // Calibrate the qualitative label using `a`'s view of the vocab; exclude `a`'s own
  // tokens so trivial self-cos = 1.0 doesn't skew the mean.
  const stats = cosineStatsAgainstMatrix(a.vector, ctx.matrix, {
    exclude: new Set<number>(a.tokenIds),
  });
  const { z, band } = classifyCosine(cos, stats);
  if (opts.json) {
    console.log(
      JSON.stringify({
        a: opts.a,
        b: opts.b,
        cosine: cos,
        z: jsonSafeZ(z),
        band,
        referenceStats: { mean: stats.mean, std: stats.std, count: stats.count },
      }),
    );
    return;
  }
  console.log(`a: ${a.label}`);
  console.log(`b: ${b.label}`);
  console.log(
    `cosine: ${formatFloat(cos)}  (${describeSimilarityBand(band)}, z=${formatZ(z)} vs reference μ=${formatFloat(stats.mean)} σ=${formatFloat(stats.std)})`,
  );
}

type ExpressionOpts = {
  readonly modelArg: string;
  readonly tokenizerPath: string;
  readonly expression: string;
  readonly topK: number;
  readonly excludeInputs: boolean;
  readonly json: boolean;
  readonly autoSpace: boolean;
};

/**
 * Apply the auto-prefix rule to an expression atom: bare identifiers get a
 * leading space (matching the GPT-2 pretokenizer convention), quoted phrases
 * keep their literal value (the explicit opt-out).
 */
function literalForAtom(value: string, quoted: boolean, autoSpace: boolean): string {
  if (quoted) return value;
  return applyAutoSpace(value, autoSpace);
}

export async function runEmbeddingsExpression(opts: ExpressionOpts): Promise<void> {
  const ctx = await loadEmbeddingsContext(opts);
  const ast = parseEmbeddingExpression(opts.expression);

  const atomNodes = collectAtomNodes(ast);
  const seenAtomKeys = new Set<string>();
  const inputIds = new Set<number>();
  type AtomTrace = {
    atom: string;
    quoted: boolean;
    encoded: string;
    ids: number[];
    surfaces: string[];
  };
  const atomTrace: AtomTrace[] = [];
  for (const node of atomNodes) {
    const key = `${node.quoted ? 'q' : 'i'}:${node.value}`;
    if (seenAtomKeys.has(key)) continue;
    seenAtomKeys.add(key);
    const encoded = literalForAtom(node.value, node.quoted, opts.autoSpace);
    const { ids, surfaces } = ctx.tokenize(encoded);
    for (const id of ids) inputIds.add(id);
    atomTrace.push({ atom: node.value, quoted: node.quoted, encoded, ids, surfaces });
  }

  const resolveAtom: ResolveAtom = (atom, atomOpts) => {
    const encoded = literalForAtom(atom, atomOpts?.quoted === true, opts.autoSpace);
    const { ids } = ctx.tokenize(encoded);
    return lookupEmbedding(ctx.matrix, ids, { pool: 'mean' }) as Float32Array;
  };
  const vec = evaluateEmbeddingExpression(ast, resolveAtom);
  const exclude = opts.excludeInputs ? inputIds : undefined;
  const top = topKByCosine(vec, ctx.matrix, opts.topK, exclude !== undefined ? { exclude } : {});
  // Always exclude the input ids from the reference distribution so the calibrated
  // bands aren't skewed by the user's own input vectors (independent of whether
  // those tokens are visible in the result list).
  const stats = cosineStatsAgainstMatrix(vec, ctx.matrix, { exclude: inputIds });
  const labeled = labelTopK(top, stats);

  if (opts.json) {
    console.log(
      JSON.stringify({
        expression: opts.expression,
        autoSpace: opts.autoSpace,
        inputs: atomTrace.map((t) => ({
          atom: t.atom,
          quoted: t.quoted,
          encoded: t.encoded,
          tokens: t.ids.map((id, i) => ({ id, surface: t.surfaces[i] ?? '' })),
        })),
        excludeInputs: opts.excludeInputs,
        topK: opts.topK,
        referenceStats: { mean: stats.mean, std: stats.std, count: stats.count },
        results: labeled.map((e) => ({
          token: decodeId(ctx.tokenizer, e.index),
          id: e.index,
          cosine: e.cosine,
          z: jsonSafeZ(e.z),
          band: e.band,
        })),
      }),
    );
    return;
  }
  console.log(`expression: ${opts.expression}`);
  for (const t of atomTrace) {
    const list = t.ids
      .map((id, i) => `${String(id)}=${JSON.stringify(t.surfaces[i] ?? '')}`)
      .join(', ');
    const encodedNote = t.encoded === t.atom ? '' : ` → encoded as ${JSON.stringify(t.encoded)}`;
    console.log(
      `  ${JSON.stringify(t.atom)}${encodedNote} → ${String(t.ids.length)} token(s): ${list}${t.ids.length > 1 ? ' [mean-pooled]' : ''}`,
    );
  }
  console.log(
    `reference cosine distribution: μ=${formatFloat(stats.mean)} σ=${formatFloat(stats.std)} (n=${String(stats.count)} rows)`,
  );
  console.log(formatTopKTable(labeled, ctx.tokenizer));
}

type NearestOpts = {
  readonly modelArg: string;
  readonly tokenizerPath: string;
  readonly word: string;
  readonly topK: number;
  readonly json: boolean;
  readonly autoSpace: boolean;
};

export async function runEmbeddingsNearest(opts: NearestOpts): Promise<void> {
  await runEmbeddingsDecode({
    modelArg: opts.modelArg,
    tokenizerPath: opts.tokenizerPath,
    vectorSpec: opts.word,
    topK: opts.topK,
    json: opts.json,
    autoSpace: opts.autoSpace,
  });
}

function emitTopK(args: {
  readonly inputs: ReadonlyArray<string>;
  readonly topK: ReadonlyArray<LabeledTopKEntry>;
  readonly stats: CosineStats;
  readonly json: boolean;
  readonly tokenizer: BPETokenizer;
}): void {
  if (args.json) {
    console.log(
      JSON.stringify({
        inputs: args.inputs,
        referenceStats: { mean: args.stats.mean, std: args.stats.std, count: args.stats.count },
        results: args.topK.map((e) => ({
          token: decodeId(args.tokenizer, e.index),
          id: e.index,
          cosine: e.cosine,
          z: jsonSafeZ(e.z),
          band: e.band,
        })),
      }),
    );
    return;
  }
  for (const inp of args.inputs) console.log(`input: ${inp}`);
  console.log(
    `reference cosine distribution: μ=${formatFloat(args.stats.mean)} σ=${formatFloat(args.stats.std)} (n=${String(args.stats.count)} rows)`,
  );
  console.log(formatTopKTable(args.topK, args.tokenizer));
}

/** Small helper exposed for tests: format a top-K list as the default aligned table. */
export const __formatTopKTableForTests = formatTopKTable;
