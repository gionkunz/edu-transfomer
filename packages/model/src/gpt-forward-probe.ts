import {
  add,
  createLeafVariable,
  embedding,
  gelu,
  matmul,
  rmsnorm,
  scale,
  softmax,
  transpose,
  type Variable,
} from '@llm-transformer-diy/autograd';
import {
  createIndices,
  floatDtypeOf,
  toArray,
  type Tensor,
  type TensorRuntime,
} from '@llm-transformer-diy/tensor';

import { buildCausalBiasTensor, type GptModel, type TransformerBlockParams } from './gpt';

export type HiddenRowScan = {
  readonly row: number;
  readonly nonFinite: number;
  readonly maxAbs: number;
};

export type ForwardProbeStage = {
  readonly name: string;
  readonly rows: readonly HiddenRowScan[];
};

export type ForwardProbeReport = {
  readonly seqLen: number;
  readonly dModel: number;
  readonly stages: readonly ForwardProbeStage[];
  /** First stage (in order) where any probed row has non-finite values. */
  readonly firstBadStage: string | null;
  /** Optional per-column detail (e.g. FFN GELU pre-activation at bad columns). */
  readonly extraLines: readonly string[];
};

/** Must match `geluScalar` / elementwise-gelu.wgsl. */
const GELU_TANH_COEF = 0.7978845608028654;
const GELU_CUBE_COEF = 0.044715;
const GELU_LINEAR_THRESHOLD = 8;

function geluScalarCpu(x: number): number {
  if (x > GELU_LINEAR_THRESHOLD) {
    return x;
  }
  if (x < -GELU_LINEAR_THRESHOLD) {
    return 0;
  }
  const t = Math.tanh(GELU_TANH_COEF * (x + GELU_CUBE_COEF * x * x * x));
  return 0.5 * x * (1 + t);
}

/** Layer index from stage names like `L7_out`, `L7_ffn_gelu`, … */
export function layerIndexFromProbeStage(stage: string): number | null {
  const m = /^L(\d+)_/.exec(stage);
  if (m === null) {
    return null;
  }
  const n = Number(m[1]);
  return Number.isInteger(n) ? n : null;
}

function formatFfnGeluColumnLines(
  lin1Host: Float32Array,
  geluHost: Float32Array,
  row: number,
  dFf: number,
  stageLabel: string,
  maxCols = 8,
): string[] {
  const base = row * dFf;
  const badCols: number[] = [];
  for (let col = 0; col < dFf; col++) {
    const g = geluHost[base + col];
    if (g === undefined || !Number.isFinite(g)) {
      badCols.push(col);
    }
  }
  if (badCols.length === 0) {
    return [];
  }
  const lines: string[] = [
    `  ${stageLabel} column detail (row ${String(row)}, ${String(badCols.length)} bad col(s)):`,
  ];
  for (const col of badCols.slice(0, maxCols)) {
    const pre = lin1Host[base + col];
    const gpu = geluHost[base + col];
    const preFinite = pre !== undefined && Number.isFinite(pre);
    const cpuOut = preFinite ? geluScalarCpu(pre) : Number.NaN;
    lines.push(
      `    col ${String(col)}: lin1=${preFinite ? pre.toExponential(6) : String(pre)}` +
        ` · gpu_gelu=${String(gpu)} · cpu_gelu=${Number.isFinite(cpuOut) ? cpuOut.toExponential(6) : String(cpuOut)}`,
    );
  }
  if (badCols.length > maxCols) {
    lines.push(`    … and ${String(badCols.length - maxCols)} more bad column(s)`);
  }
  return lines;
}

function scanHiddenRows(
  host: Float32Array,
  seqLen: number,
  cols: number,
  rows: readonly number[],
): HiddenRowScan[] {
  const out: HiddenRowScan[] = [];
  for (const row of rows) {
    if (row < 0 || row >= seqLen) {
      continue;
    }
    const base = row * cols;
    let nonFinite = 0;
    let maxAbs = 0;
    for (let j = 0; j < cols; j++) {
      const v = host[base + j];
      if (v === undefined || !Number.isFinite(v)) {
        nonFinite++;
        continue;
      }
      const a = Math.abs(v);
      if (a > maxAbs) {
        maxAbs = a;
      }
    }
    out.push({ row, nonFinite, maxAbs });
  }
  return out;
}

async function stageFromVariable(
  rt: TensorRuntime,
  name: string,
  h: Variable,
  seqLen: number,
  cols: number,
  rows: readonly number[],
): Promise<ForwardProbeStage> {
  const host = await toArray(rt, h.tensor);
  return { name, rows: scanHiddenRows(host, seqLen, cols, rows) };
}

function mhaForwardProbe(
  rt: TensorRuntime,
  blk: TransformerBlockParams,
  x: Variable,
  maskVar: Variable,
  invSqrtDk: number,
  headExpand: readonly Variable[],
): Variable {
  const H = blk.wq.length;
  let acc: Variable | null = null;
  for (let h = 0; h < H; h++) {
    const wqh = blk.wq[h];
    const wkh = blk.wk[h];
    const wvh = blk.wv[h];
    const he = headExpand[h];
    if (!wqh || !wkh || !wvh || !he) {
      throw new Error('mhaForwardProbe: missing head tensors');
    }
    const q = matmul(rt, x, wqh);
    const k = matmul(rt, x, wkh);
    const vMat = matmul(rt, x, wvh);
    const kt = transpose(rt, k);
    let scores = matmul(rt, q, kt);
    scores = scale(rt, scores, invSqrtDk);
    scores = add(rt, scores, maskVar);
    const prob = softmax(rt, scores);
    const mixed = matmul(rt, prob, vMat);
    const contrib = matmul(rt, mixed, he);
    acc = acc === null ? contrib : add(rt, acc, contrib);
  }
  if (acc === null) {
    throw new Error('mhaForwardProbe: nHeads must be >= 1');
  }
  return matmul(rt, acc, blk.wo);
}

async function blockForwardProbeDetailed(
  rt: TensorRuntime,
  blk: TransformerBlockParams,
  layerIndex: number,
  x: Variable,
  maskVar: Variable,
  invSqrtDk: number,
  headExpand: readonly Variable[],
  seqLen: number,
  dModel: number,
  dFf: number,
  rows: readonly number[],
  stages: ForwardProbeStage[],
  extraLines: string[],
): Promise<Variable> {
  const p = `L${String(layerIndex)}`;

  let y = rmsnorm(rt, x, blk.attnNormGamma);
  stages.push(await stageFromVariable(rt, `${p}_attn_norm`, y, seqLen, dModel, rows));

  const attn = mhaForwardProbe(rt, blk, y, maskVar, invSqrtDk, headExpand);
  stages.push(await stageFromVariable(rt, `${p}_attn`, attn, seqLen, dModel, rows));

  x = add(rt, x, attn);
  stages.push(await stageFromVariable(rt, `${p}_post_attn`, x, seqLen, dModel, rows));

  y = rmsnorm(rt, x, blk.ffnNormGamma);
  stages.push(await stageFromVariable(rt, `${p}_ffn_norm`, y, seqLen, dModel, rows));

  let ff = matmul(rt, y, blk.w1);
  const lin1Host = await toArray(rt, ff.tensor);
  stages.push({ name: `${p}_ffn_lin1`, rows: scanHiddenRows(lin1Host, seqLen, dFf, rows) });

  ff = gelu(rt, ff);
  const geluHost = await toArray(rt, ff.tensor);
  stages.push({ name: `${p}_ffn_gelu`, rows: scanHiddenRows(geluHost, seqLen, dFf, rows) });
  for (const row of rows) {
    const rs = scanHiddenRows(geluHost, seqLen, dFf, [row])[0];
    if (rs !== undefined && rs.nonFinite > 0) {
      extraLines.push(...formatFfnGeluColumnLines(lin1Host, geluHost, row, dFf, `${p}_ffn_gelu`));
    }
  }

  ff = matmul(rt, ff, blk.w2);
  stages.push(await stageFromVariable(rt, `${p}_ffn_lin2`, ff, seqLen, dModel, rows));

  x = add(rt, x, ff);
  stages.push(await stageFromVariable(rt, `${p}_out`, x, seqLen, dModel, rows));
  return x;
}

/**
 * Runs the same forward path as {@link forwardGpt} and readbacks hidden rows after each sub-layer.
 * Does not build a backward graph past the returned logits (caller should not backward on these nodes
 * without registering parameters — intended for diagnostics only).
 */
export async function probeGptForward(
  rt: TensorRuntime,
  model: GptModel,
  tokenIds: Tensor,
  rowsToScan: readonly number[],
  opts?: { readonly detailedBlock?: number },
): Promise<{ readonly report: ForwardProbeReport; readonly logits: Variable }> {
  const T = tokenIds.shape[0];
  if (T === undefined) {
    throw new Error('probeGptForward: tokenIds must be non-empty');
  }
  const dModel = model.config.dModel;
  const dFf = model.config.dFf;
  const stages: ForwardProbeStage[] = [];
  const extraLines: string[] = [];

  const tokIdxVar = createLeafVariable(tokenIds, false);
  const posHost = new Uint32Array(T);
  for (let i = 0; i < T; i++) {
    posHost[i] = i;
  }
  const posTensor = createIndices(rt, posHost, [T]);
  const posVar = createLeafVariable(posTensor, false);

  let h = add(
    rt,
    embedding(rt, model.tokenEmbed, tokIdxVar),
    embedding(rt, model.posEmbed, posVar),
  );
  stages.push(await stageFromVariable(rt, 'embed', h, T, dModel, rowsToScan));

  const maskTensor = buildCausalBiasTensor(rt, T, -1e4, floatDtypeOf(model.tokenEmbed.tensor));
  const maskVar = createLeafVariable(maskTensor, false);
  const dK = model.config.dModel / model.config.nHeads;
  const invSqrtDk = 1 / Math.sqrt(dK);

  for (let li = 0; li < model.blocks.length; li++) {
    const blk = model.blocks[li];
    if (!blk) {
      throw new Error('probeGptForward: missing block');
    }
    if (opts?.detailedBlock === li) {
      h = await blockForwardProbeDetailed(
        rt,
        blk,
        li,
        h,
        maskVar,
        invSqrtDk,
        model.headExpand,
        T,
        dModel,
        dFf,
        rowsToScan,
        stages,
        extraLines,
      );
    } else {
      let y = rmsnorm(rt, h, blk.attnNormGamma);
      const attn = mhaForwardProbe(rt, blk, y, maskVar, invSqrtDk, model.headExpand);
      h = add(rt, h, attn);
      y = rmsnorm(rt, h, blk.ffnNormGamma);
      let ff = matmul(rt, y, blk.w1);
      ff = gelu(rt, ff);
      ff = matmul(rt, ff, blk.w2);
      h = add(rt, h, ff);
      stages.push(await stageFromVariable(rt, `L${String(li)}_out`, h, T, dModel, rowsToScan));
    }
  }

  h = rmsnorm(rt, h, model.finalNormGamma);
  stages.push(await stageFromVariable(rt, 'final_norm', h, T, dModel, rowsToScan));

  const logits = matmul(rt, h, model.lmHead);
  const logitHost = await toArray(rt, logits.tensor);
  const vocabSize = model.config.vocabSize;
  stages.push({
    name: 'logits',
    rows: scanHiddenRows(logitHost, T, vocabSize, rowsToScan),
  });

  let firstBadStage: string | null = null;
  for (const st of stages) {
    if (st.rows.some((r) => r.nonFinite > 0)) {
      firstBadStage = st.name;
      break;
    }
  }

  return {
    report: { seqLen: T, dModel, stages, firstBadStage, extraLines },
    logits,
  };
}

/** Formats a {@link ForwardProbeReport} for stderr (multi-line, indented). */
export function formatForwardProbeReport(
  report: ForwardProbeReport,
  rowsToScan: readonly number[],
): string[] {
  const lines: string[] = ['  forward probe (hidden rows):'];
  for (const row of rowsToScan) {
    lines.push(`    row ${String(row)}:`);
    for (const st of report.stages) {
      const rs = st.rows.find((r) => r.row === row);
      if (!rs) {
        continue;
      }
      const flag = rs.nonFinite > 0 ? ' NON-FINITE' : '';
      const denom = st.name === 'logits' ? 'vocab' : 'd';
      lines.push(
        `      ${st.name}: ${String(rs.nonFinite)} non-finite · max|${denom}|=${rs.maxAbs.toExponential(3)}${flag}`,
      );
    }
  }
  if (report.firstBadStage !== null) {
    lines.push(`  first non-finite stage: ${report.firstBadStage}`);
  }
  lines.push(...report.extraLines);
  return lines;
}

/** Corpus batch offset for a training step (matches `llm train` loop). */
export function trainingBatchStart(tokenCount: number, seqLen: number, step: number): number {
  const maxStart = tokenCount - seqLen - 1;
  if (maxStart < 0) {
    throw new Error('trainingBatchStart: corpus too short for seqLen');
  }
  return Math.floor((step * 9973) % (maxStart + 1));
}

/** Reads **`count`** token ids from a little-endian `.ids` cache file at **`start`**. */
export async function readTokenIdsSliceFromFile(
  idsPath: string,
  start: number,
  count: number,
): Promise<Uint32Array> {
  const { open } = await import('node:fs/promises');
  const buf = Buffer.alloc(count * 4);
  const fh = await open(idsPath, 'r');
  try {
    await fh.read(buf, 0, count * 4, start * 4);
  } finally {
    await fh.close();
  }
  return new Uint32Array(buf.buffer, buf.byteOffset, count);
}
