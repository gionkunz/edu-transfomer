import {
  formatForwardProbeReport,
  probeGptForward,
  trainingBatchStart,
  type GptModel,
} from '@llm-transformer-diy/model';
import { destroy, readIndices, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';

import { createLmBatch } from './lm-batch';
import { scanLogitsRowsByPosition } from './train-failure-diagnostics';
import type { Trainer } from './trainer';

/** Runs one training step index, then a forward layer probe, printing to stderr. */
export async function runReplayStepProbe(opts: {
  readonly rt: TensorRuntime;
  readonly model: GptModel;
  readonly trainer: Trainer;
  readonly tokenIds: Uint32Array;
  readonly seqLen: number;
  readonly vocabSize: number;
  readonly fromStep: number;
  readonly replayStep: number;
  readonly rowsToScan?: readonly number[];
}): Promise<void> {
  const { rt, model, trainer, tokenIds, seqLen, vocabSize, fromStep, replayStep } = opts;

  if (!Number.isInteger(replayStep) || replayStep < 0) {
    throw new Error(`runReplayStepProbe: replayStep must be a non-negative integer`);
  }
  if (fromStep > replayStep) {
    throw new Error(
      `runReplayStepProbe: fromStep ${String(fromStep)} > replayStep ${String(replayStep)}`,
    );
  }

  for (let s = fromStep; s < replayStep; s++) {
    const start = trainingBatchStart(tokenIds.length, seqLen, s);
    const { input, targets } = createLmBatch(rt, tokenIds, start, seqLen);
    try {
      await trainer.trainStep(input, targets, s, { batchStart: start, vocabSize });
    } finally {
      destroy(input);
      destroy(targets);
    }
    if ((s + 1) % 500 === 0 || s + 1 === replayStep) {
      console.error(`train replay: completed step ${String(s + 1)}/${String(replayStep)}`);
    }
  }

  const batchStart = trainingBatchStart(tokenIds.length, seqLen, replayStep);
  const { input, targets } = createLmBatch(rt, tokenIds, batchStart, seqLen);
  try {
    console.error(
      `train replay: probing step ${String(replayStep)} batch start=${String(batchStart)}`,
    );
    const inputHost = await readIndices(rt, input);
    const defaultRows = opts.rowsToScan ?? [0, Math.floor(seqLen / 2), seqLen - 1];
    const rowsToScan = [...new Set(defaultRows)].sort((a, b) => a - b);

    let { report, logits } = await probeGptForward(rt, model, input, rowsToScan);
    const lines = [
      `train replay: forward probe at step ${String(replayStep)} (weights after ${String(replayStep)} optimizer steps from resume point):`,
      ...formatForwardProbeReport(report, rowsToScan),
    ];

    const logitHost = await toArray(rt, logits.tensor);
    const badRows = scanLogitsRowsByPosition(logitHost, seqLen, vocabSize);
    if (badRows.length > 0) {
      lines.push(`  logit rows with NaN: ${badRows.map((r) => String(r.row)).join(', ')}`);
    }

    const layerMatch = /^L(\d+)_out$/.exec(report.firstBadStage ?? '');
    if (layerMatch !== null) {
      const layerIndex = Number(layerMatch[1]);
      if (Number.isInteger(layerIndex)) {
        lines.push(`  expanding block ${String(layerIndex)}…`);
        const detailed = await probeGptForward(rt, model, input, rowsToScan, {
          detailedBlock: layerIndex,
        });
        lines.push(...formatForwardProbeReport(detailed.report, rowsToScan));
        report = detailed.report;
      }
    }

    for (const row of rowsToScan) {
      const tid = inputHost[row];
      if (tid !== undefined) {
        lines.push(`  row ${String(row)} input token id=${String(tid)}`);
      }
    }

    console.error(lines.join('\n'));
    if (report.firstBadStage !== null) {
      console.error(`train replay: first non-finite stage = ${report.firstBadStage}`);
    } else if (badRows.length === 0) {
      console.error(
        'train replay: all probed rows finite at logits (unexpected if training NaNs here)',
      );
    }
  } finally {
    destroy(input);
    destroy(targets);
  }
}
