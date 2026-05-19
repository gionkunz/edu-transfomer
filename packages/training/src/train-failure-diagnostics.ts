import type { Variable } from '@llm-transformer-diy/autograd';
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

function classifyFailure(oobCount: number, logitNonFinite: number, logitCount: number): string {
  if (oobCount > 0) {
    return 'out-of-vocab target id(s) in this batch (CE reads past logits row end)';
  }
  if (logitNonFinite > 0) {
    const ratio = logitNonFinite / Math.max(1, logitCount);
    if (ratio >= 0.5) {
      return 'most logits non-finite — weights/activations likely exploded before CE';
    }
    return 'some logits non-finite — localized numerical blow-up in forward pass';
  }
  return 'logits/targets look in-range; CE output non-finite despite finite inputs (rare kernel edge case or partial GPU corruption — retry step)';
}

/**
 * Builds a multi-line stderr block with batch offset, target bounds, and logit stats.
 * Only called on the failure path (one GPU readback of logits + targets).
 */
export async function buildNonFiniteLossDiagnostics(
  rt: TensorRuntime,
  opts: {
    readonly trainingStepIndex: number;
    readonly kind: string;
    readonly lr: number;
    readonly logits: Variable;
    readonly targets: Tensor;
    readonly batch?: TrainStepBatchContext;
    readonly seqLen: number;
  },
): Promise<string> {
  const { trainingStepIndex, kind, lr, logits, targets, batch, seqLen } = opts;
  const lines: string[] = [
    `trainer: non-finite loss at step ${String(trainingStepIndex)} (${kind}).`,
  ];

  if (batch !== undefined) {
    lines.push(
      `  batch: corpus offset start=${String(batch.batchStart)} seqLen=${String(seqLen)} ` +
        `(tokens [${String(batch.batchStart)}..${String(batch.batchStart + seqLen)}))`,
    );
  }

  lines.push(`  lr=${String(lr)} (schedule index ${String(trainingStepIndex)})`);

  let targetStats = { maxTarget: 0, oobCount: 0 };
  try {
    const targetHost = await readIndices(rt, targets);
    targetStats = scanTargetStats(targetHost, batch?.vocabSize ?? Number.POSITIVE_INFINITY);
    const vocabLabel =
      batch?.vocabSize !== undefined ? String(batch.vocabSize) : 'unknown (pass batch.vocabSize)';
    lines.push(
      `  targets: max id=${String(targetStats.maxTarget)} vocabSize=${vocabLabel}` +
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
  try {
    const logitHost = await toArray(rt, logits.tensor);
    logitCount = logitHost.length;
    logitStats = scanFloat32FiniteStats(logitHost);
    lines.push(
      `  logits: ${String(logitStats.nonFinite)}/${String(logitCount)} non-finite` +
        ` · max|logit|=${Number.isFinite(logitStats.maxAbs) ? logitStats.maxAbs.toExponential(3) : 'n/a'}` +
        (Number.isFinite(logitStats.min) && Number.isFinite(logitStats.max)
          ? ` · range=[${logitStats.min.toExponential(2)}, ${logitStats.max.toExponential(2)}]`
          : ''),
    );
  } catch (e) {
    lines.push(`  logits: readback failed (${e instanceof Error ? e.message : String(e)})`);
  }

  const verdict = classifyFailure(targetStats.oobCount, logitStats.nonFinite, logitCount);
  lines.push(`  likely cause: ${verdict}`);

  if (targetStats.oobCount > 0) {
    lines.push(
      '  fix: delete token cache, re-encode with --rebuild-token-cache, confirm tokenizer vocab matches config',
    );
  } else if (logitStats.nonFinite > 0) {
    lines.push(
      '  fix: halve --lr-max (you may still be at/near peak LR after warmup), try --dtype bf16, or tighten --max-grad-norm',
    );
  } else {
    lines.push('  fix: retry the run; if it repeats at the same step, report batch offset above');
  }

  return lines.join('\n');
}
