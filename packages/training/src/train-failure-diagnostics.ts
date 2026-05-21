import type { Parameter, Variable } from '@llm-transformer-diy/autograd';
import {
  collectGptParameters,
  formatForwardProbeReport,
  gptParameterTensorNames,
  layerIndexFromProbeStage,
  probeGptForward,
  type GptModel,
} from '@llm-transformer-diy/model';
import { readIndices, toArray, type Tensor, type TensorRuntime } from '@llm-transformer-diy/tensor';

/** Optional batch metadata so a non-finite loss can be replayed from the corpus. */
export type TrainStepBatchContext = {
  readonly batchStart: number;
  readonly vocabSize: number;
};

export type Float32FiniteScan = {
  readonly nonFinite: number;
  readonly maxAbs: number;
  readonly min: number;
  readonly max: number;
};

export type LogitsRowScan = {
  readonly row: number;
  readonly nonFiniteInRow: number;
  readonly finiteInRow: number;
  readonly maxAbsFinite: number;
};

export type TokenPositionDiagnostic = {
  readonly row: number;
  readonly inputTokenId: number;
  readonly targetTokenId: number;
  readonly corpusInputIndex: number;
  readonly corpusTargetIndex: number;
  readonly logits: LogitsRowScan;
};

export type EmbeddingRowScan = {
  readonly tokenId: number;
  readonly nonFinite: number;
  readonly maxAbs: number;
};

export type ParameterTensorAudit = {
  readonly name: string;
  readonly elements: number;
  readonly nonFinite: number;
  readonly maxAbs: number;
};

export type ParameterFiniteAudit = {
  readonly totalElements: number;
  readonly totalNonFinite: number;
  readonly tensors: readonly ParameterTensorAudit[];
};

/** CPU-side scan of a logits buffer (used after GPU readback on failure). */
export function scanFloat32FiniteStats(data: Float32Array): Float32FiniteScan {
  let nonFinite = 0;
  let maxAbs = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v === undefined || !Number.isFinite(v)) {
      nonFinite++;
      continue;
    }
    const a = Math.abs(v);
    if (a > maxAbs) {
      maxAbs = a;
    }
    if (v < min) {
      min = v;
    }
    if (v > max) {
      max = v;
    }
  }
  if (nonFinite === data.length) {
    return { nonFinite, maxAbs: Number.NaN, min: Number.NaN, max: Number.NaN };
  }
  if (nonFinite > 0) {
    if (!Number.isFinite(min)) {
      min = Number.NaN;
    }
    if (!Number.isFinite(max)) {
      max = Number.NaN;
    }
  }
  return { nonFinite, maxAbs, min, max };
}

/**
 * Row-wise scan of LM logits **`[seqLen, vocabSize]`** in row-major layout.
 * Returns rows with any non-finite logit, sorted by row index.
 */
export function scanLogitsRowsByPosition(
  logitHost: Float32Array,
  seqLen: number,
  vocabSize: number,
): LogitsRowScan[] {
  const expected = seqLen * vocabSize;
  if (logitHost.length !== expected) {
    throw new Error(
      `scanLogitsRowsByPosition: length ${String(logitHost.length)} ≠ seqLen×vocab (${String(expected)})`,
    );
  }
  const bad: LogitsRowScan[] = [];
  for (let row = 0; row < seqLen; row++) {
    const base = row * vocabSize;
    let nonFiniteInRow = 0;
    let finiteInRow = 0;
    let maxAbsFinite = 0;
    for (let j = 0; j < vocabSize; j++) {
      const v = logitHost[base + j];
      if (v === undefined || !Number.isFinite(v)) {
        nonFiniteInRow++;
        continue;
      }
      finiteInRow++;
      const a = Math.abs(v);
      if (a > maxAbsFinite) {
        maxAbsFinite = a;
      }
    }
    if (nonFiniteInRow > 0) {
      bad.push({ row, nonFiniteInRow, finiteInRow, maxAbsFinite });
    }
  }
  return bad;
}

/** Map bad logit rows to input/target token ids and absolute corpus indices. */
export function correlateBadRowsWithTokens(
  badRows: readonly LogitsRowScan[],
  inputIds: Uint32Array,
  targetIds: Uint32Array,
  batchStart: number,
): TokenPositionDiagnostic[] {
  const out: TokenPositionDiagnostic[] = [];
  for (const logits of badRows) {
    const row = logits.row;
    const inputTokenId = inputIds[row];
    const targetTokenId = targetIds[row];
    if (inputTokenId === undefined || targetTokenId === undefined) {
      continue;
    }
    out.push({
      row,
      inputTokenId,
      targetTokenId,
      corpusInputIndex: batchStart + row,
      corpusTargetIndex: batchStart + row + 1,
      logits,
    });
  }
  return out;
}

function scanTargetStats(
  targets: Uint32Array,
  vocabSize: number,
): {
  readonly maxTarget: number;
  readonly oobCount: number;
} {
  let maxTarget = 0;
  let oobCount = 0;
  for (let i = 0; i < targets.length; i++) {
    const t = targets[i];
    if (t === undefined) {
      continue;
    }
    if (t > maxTarget) {
      maxTarget = t;
    }
    if (t >= vocabSize) {
      oobCount++;
    }
  }
  return { maxTarget, oobCount };
}

/** Read **`token_embed`** rows for the given ids (shape `[vocab, d_model]`). */
export async function scanTokenEmbeddingRows(
  rt: TensorRuntime,
  tokenEmbed: Parameter,
  tokenIds: readonly number[],
): Promise<EmbeddingRowScan[]> {
  const shape = tokenEmbed.tensor.shape;
  const vocabDim = shape[0];
  const dModel = shape[1];
  if (vocabDim === undefined || dModel === undefined) {
    throw new Error('scanTokenEmbeddingRows: token_embed must be rank-2');
  }
  const full = await toArray(rt, tokenEmbed.tensor);
  const unique = [...new Set(tokenIds)].sort((a, b) => a - b);
  const out: EmbeddingRowScan[] = [];
  for (const tokenId of unique) {
    if (tokenId < 0 || tokenId >= vocabDim) {
      out.push({ tokenId, nonFinite: -1, maxAbs: Number.NaN });
      continue;
    }
    const base = tokenId * dModel;
    let nonFinite = 0;
    let maxAbs = 0;
    for (let j = 0; j < dModel; j++) {
      const v = full[base + j];
      if (v === undefined || !Number.isFinite(v)) {
        nonFinite++;
        continue;
      }
      const a = Math.abs(v);
      if (a > maxAbs) {
        maxAbs = a;
      }
    }
    out.push({ tokenId, nonFinite, maxAbs });
  }
  return out;
}

/** Full GPU readback audit of every trainable tensor (use on failure or pre-resume). */
export async function auditModelParametersFinite(
  rt: TensorRuntime,
  model: GptModel,
): Promise<ParameterFiniteAudit> {
  const names = gptParameterTensorNames(model);
  const params = collectGptParameters(model);
  if (names.length !== params.length) {
    throw new Error('auditModelParametersFinite: name/param length mismatch');
  }

  let totalElements = 0;
  let totalNonFinite = 0;
  const tensors: ParameterTensorAudit[] = [];

  for (let i = 0; i < params.length; i++) {
    const name = names[i];
    const p = params[i];
    if (name === undefined || p === undefined) {
      throw new Error('auditModelParametersFinite: internal index error');
    }
    const host = await toArray(rt, p.tensor);
    const stats = scanFloat32FiniteStats(host);
    totalElements += host.length;
    totalNonFinite += stats.nonFinite;
    if (stats.nonFinite > 0) {
      tensors.push({
        name,
        elements: host.length,
        nonFinite: stats.nonFinite,
        maxAbs: stats.maxAbs,
      });
    }
  }

  tensors.sort((a, b) => b.nonFinite - a.nonFinite);
  return { totalElements, totalNonFinite, tensors };
}

function classifyFailure(
  oobCount: number,
  logitNonFinite: number,
  logitCount: number,
  badRowCount: number,
  paramNonFinite: number,
): string {
  if (oobCount > 0) {
    return 'out-of-vocab target id(s) in this batch (CE reads past logits row end)';
  }
  if (paramNonFinite > 0 && logitNonFinite === 0) {
    return 'weights already contain non-finite values before this forward pass';
  }
  if (logitNonFinite > 0) {
    const ratio = logitNonFinite / Math.max(1, logitCount);
    if (ratio >= 0.5) {
      return 'most logits non-finite — global activation/weight explosion before CE';
    }
    if (badRowCount === 1) {
      return 'single sequence position has all-NaN logits — localized forward blow-up (check token_embed row for input id below)';
    }
    if (badRowCount > 0 && badRowCount <= 8) {
      return `${String(badRowCount)} sequence position(s) have non-finite logits — localized forward blow-up`;
    }
    return 'scattered non-finite logits across many positions — likely unstable dynamics or bad weights';
  }
  return 'logits/targets look in-range; CE output non-finite despite finite inputs (rare kernel edge case — retry step)';
}

function formatEmbeddingRowLines(scans: readonly EmbeddingRowScan[]): string[] {
  const lines: string[] = [];
  for (const s of scans) {
    if (s.nonFinite < 0) {
      lines.push(`      token ${String(s.tokenId)}: OOB for token_embed`);
      continue;
    }
    if (s.nonFinite > 0) {
      lines.push(
        `      token ${String(s.tokenId)}: token_embed row ${String(s.nonFinite)} non-finite · max|w|=${Number.isFinite(s.maxAbs) ? s.maxAbs.toExponential(3) : 'n/a'}`,
      );
    } else {
      lines.push(
        `      token ${String(s.tokenId)}: token_embed row OK · max|w|=${s.maxAbs.toExponential(3)}`,
      );
    }
  }
  return lines;
}

function formatPositionLines(
  positions: readonly TokenPositionDiagnostic[],
  maxLines: number,
): string[] {
  const lines: string[] = [];
  const slice = positions.slice(0, maxLines);
  for (const p of slice) {
    const { logits: lg } = p;
    lines.push(
      `    pos ${String(p.row)}: input=${String(p.inputTokenId)} target=${String(p.targetTokenId)} ` +
        `corpus [${String(p.corpusInputIndex)}→${String(p.corpusTargetIndex)}] ` +
        `logits ${String(lg.nonFiniteInRow)}/${String(lg.nonFiniteInRow + lg.finiteInRow)} non-finite` +
        (lg.finiteInRow > 0 ? ` · max|logit|=${lg.maxAbsFinite.toExponential(3)}` : ''),
    );
  }
  if (positions.length > maxLines) {
    lines.push(`    … and ${String(positions.length - maxLines)} more bad position(s)`);
  }
  return lines;
}

/**
 * Builds a multi-line stderr block with batch offset, per-position logits, token ids,
 * embedding probes, and optional full weight audit.
 */
export async function buildNonFiniteLossDiagnostics(
  rt: TensorRuntime,
  opts: {
    readonly trainingStepIndex: number;
    readonly kind: string;
    readonly lr: number;
    readonly logits: Variable;
    readonly tokenInput: Tensor;
    readonly targets: Tensor;
    readonly batch?: TrainStepBatchContext;
    readonly seqLen: number;
    readonly model?: GptModel;
    /** When true, read back every parameter tensor (slow; default true on failure). */
    readonly auditAllParameters?: boolean;
  },
): Promise<string> {
  const {
    trainingStepIndex,
    kind,
    lr,
    logits,
    tokenInput,
    targets,
    batch,
    seqLen,
    model,
    auditAllParameters = true,
  } = opts;

  const vocabSize = batch?.vocabSize ?? logits.tensor.shape[1];
  if (vocabSize === undefined || !Number.isInteger(vocabSize) || vocabSize < 1) {
    throw new Error(
      'buildNonFiniteLossDiagnostics: vocabSize required (batch.vocabSize or logits.shape[1])',
    );
  }

  const lines: string[] = [
    `trainer: non-finite loss at step ${String(trainingStepIndex)} (${kind}).`,
  ];

  if (batch !== undefined) {
    lines.push(
      `  batch: corpus offset start=${String(batch.batchStart)} seqLen=${String(seqLen)} ` +
        `(tokens [${String(batch.batchStart)}..${String(batch.batchStart + seqLen)}))`,
    );
    lines.push(
      `  replay: same step index ${String(trainingStepIndex)} always uses this offset ` +
        `(start = step × 9973 mod corpusLen)`,
    );
  }

  lines.push(`  lr=${String(lr)} (schedule index ${String(trainingStepIndex)})`);

  let targetStats = { maxTarget: 0, oobCount: 0 };
  let inputHost = new Uint32Array(0);
  let targetHost = new Uint32Array(0);
  try {
    inputHost = await readIndices(rt, tokenInput);
    targetHost = await readIndices(rt, targets);
    targetStats = scanTargetStats(targetHost, vocabSize);
    lines.push(
      `  targets: max id=${String(targetStats.maxTarget)} vocabSize=${String(vocabSize)}` +
        (targetStats.oobCount > 0 ? ` · ${String(targetStats.oobCount)} id(s) ≥ vocab` : ''),
    );
  } catch (e) {
    lines.push(`  targets: readback failed (${e instanceof Error ? e.message : String(e)})`);
  }

  let logitStats: Float32FiniteScan = {
    nonFinite: 0,
    maxAbs: Number.NaN,
    min: Number.NaN,
    max: Number.NaN,
  };
  let logitCount = 0;
  let badRows: LogitsRowScan[] = [];
  let positions: TokenPositionDiagnostic[] = [];
  try {
    const logitHost = await toArray(rt, logits.tensor);
    logitCount = logitHost.length;
    logitStats = scanFloat32FiniteStats(logitHost);
    badRows = scanLogitsRowsByPosition(logitHost, seqLen, vocabSize);
    if (batch !== undefined && inputHost.length > 0 && targetHost.length > 0) {
      positions = correlateBadRowsWithTokens(badRows, inputHost, targetHost, batch.batchStart);
    }
    lines.push(
      `  logits: ${String(logitStats.nonFinite)}/${String(logitCount)} non-finite` +
        ` · ${String(badRows.length)}/${String(seqLen)} position(s) affected` +
        ` · max|logit|=${Number.isFinite(logitStats.maxAbs) ? logitStats.maxAbs.toExponential(3) : 'n/a'}` +
        (Number.isFinite(logitStats.min) && Number.isFinite(logitStats.max)
          ? ` · global range=[${logitStats.min.toExponential(2)}, ${logitStats.max.toExponential(2)}]`
          : ''),
    );
    if (positions.length > 0) {
      lines.push('  bad positions (input → target at each row):');
      lines.push(...formatPositionLines(positions, 12));
    }
  } catch (e) {
    lines.push(`  logits: readback failed (${e instanceof Error ? e.message : String(e)})`);
  }

  let paramNonFinite = 0;
  if (model !== undefined && auditAllParameters) {
    try {
      const audit = await auditModelParametersFinite(rt, model);
      paramNonFinite = audit.totalNonFinite;
      lines.push(
        `  weights: ${String(audit.totalNonFinite)}/${String(audit.totalElements)} non-finite across ${String(audit.tensors.length)} tensor(s)`,
      );
      const top = audit.tensors.slice(0, 6);
      for (const t of top) {
        lines.push(
          `    ${t.name}: ${String(t.nonFinite)}/${String(t.elements)} non-finite · max|w|=${Number.isFinite(t.maxAbs) ? t.maxAbs.toExponential(3) : 'n/a'}`,
        );
      }
      if (audit.tensors.length > top.length) {
        lines.push(
          `    … and ${String(audit.tensors.length - top.length)} more tensor(s) with non-finite values`,
        );
      }
    } catch (e) {
      lines.push(`  weights: audit failed (${e instanceof Error ? e.message : String(e)})`);
    }
  }

  if (model !== undefined && positions.length > 0) {
    const probeIds = positions.map((p) => p.inputTokenId);
    try {
      lines.push('  token_embed rows for input ids at bad positions:');
      const embedScans = await scanTokenEmbeddingRows(rt, model.tokenEmbed, probeIds);
      lines.push(...formatEmbeddingRowLines(embedScans));
    } catch (e) {
      lines.push(`  token_embed probe failed (${e instanceof Error ? e.message : String(e)})`);
    }

    try {
      const rowsToScan = [...new Set(positions.map((p) => p.row))].sort((a, b) => a - b);
      lines.push('  running forward layer probe (re-forward, no backward)…');
      let { report } = await probeGptForward(rt, model, tokenInput, rowsToScan);
      lines.push(...formatForwardProbeReport(report, rowsToScan));

      const layerIndex = layerIndexFromProbeStage(report.firstBadStage ?? '');
      const alreadyDetailed =
        layerIndex !== null &&
        report.stages.some((s) => s.name === `L${String(layerIndex)}_ffn_lin1`);
      if (layerIndex !== null && !alreadyDetailed) {
        lines.push(`  expanding probe inside block ${String(layerIndex)} (attn/ffn substeps)…`);
        const detailed = await probeGptForward(rt, model, tokenInput, rowsToScan, {
          detailedBlock: layerIndex,
        });
        lines.push(...formatForwardProbeReport(detailed.report, rowsToScan));
        report = detailed.report;
      }
    } catch (e) {
      lines.push(`  forward layer probe failed (${e instanceof Error ? e.message : String(e)})`);
    }
  }

  const verdict = classifyFailure(
    targetStats.oobCount,
    logitStats.nonFinite,
    logitCount,
    badRows.length,
    paramNonFinite,
  );
  lines.push(`  likely cause: ${verdict}`);

  if (targetStats.oobCount > 0) {
    lines.push(
      '  fix: delete token cache, re-encode with --rebuild-token-cache, confirm tokenizer vocab matches config',
    );
  } else if (paramNonFinite > 0 && logitStats.nonFinite === 0) {
    lines.push(
      '  fix: do not --resume this checkpoint — weights are corrupt; restart from earlier checkpoint or fresh init',
    );
  } else if (logitStats.nonFinite > 0) {
    lines.push(
      '  fix: lower --lr-max / lengthen --warmup / tighten --max-grad-norm; if one position repeats, inspect corpus at corpusInputIndex',
    );
    if (batch !== undefined && positions.length === 1) {
      const p = positions[0];
      if (p !== undefined) {
        lines.push(
          `  probe: decode corpus ids [${String(p.corpusInputIndex - 2)}..${String(p.corpusTargetIndex + 2)}] around the bad token`,
        );
      }
    }
  } else {
    lines.push('  fix: retry the run; if it repeats at the same step, report batch offset above');
  }

  return lines.join('\n');
}

/**
 * Pre-flight check after loading a checkpoint (stderr). Throws if weights contain non-finite values.
 */
export async function assertModelParametersFiniteForResume(
  rt: TensorRuntime,
  model: GptModel,
  label: string,
): Promise<void> {
  console.error(`train: auditing checkpoint weights (${label})…`);
  const audit = await auditModelParametersFinite(rt, model);
  if (audit.totalNonFinite === 0) {
    console.error(
      `train: checkpoint weights OK (${String(audit.totalElements)} elements, all finite)`,
    );
    return;
  }
  const lines: string[] = [
    `train: checkpoint "${label}" has non-finite weights — refusing to --resume.`,
    `  ${String(audit.totalNonFinite)}/${String(audit.totalElements)} non-finite element(s) in ${String(audit.tensors.length)} tensor(s):`,
  ];
  for (const t of audit.tensors.slice(0, 8)) {
    lines.push(`    ${t.name}: ${String(t.nonFinite)}/${String(t.elements)} non-finite`);
  }
  throw new Error(lines.join('\n'));
}
