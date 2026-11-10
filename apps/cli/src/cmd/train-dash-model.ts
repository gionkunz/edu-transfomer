import type { CosineLrSchedule } from '@llm-transformer-diy/training';

import { MetricSeries } from './metric-series';
import type { TrainDashboardSnapshot, TrainPipelinePhase, ValLossPoint } from './train-dash-types';

/** Mutable authoring surface consumed by **`finalizeTrainDashboardDraft`**. */
export interface TrainDashboardDraft {
  phase: TrainPipelinePhase;
  detailLine: string;
  startedAtMs: number;
  outDirDisplay: string;
  configTag: string;
  archSummary: string | null;
  gpuAdapterSummary: string | null;
  gpuSoftwareWarning: string | null;

  fileProgress: TrainDashboardSnapshot['fileProgress'];
  encodeProgress: TrainDashboardSnapshot['encodeProgress'];
  saveProgress: TrainDashboardSnapshot['saveProgress'];

  corpusChars: number | null;
  corpusTokens: number | null;
  seqLen: number;
  totalTrainSteps: number;

  trainStepCompleted: number;
  trainLossLast: number;
  trainLrDisplay: number;
  tokensPerSecApprox: number;
  recentLosses: number[];
  lossHistory: MetricSeries;
  gradNormHistory: MetricSeries;
  valHistory: ValLossPoint[];
  gradNormLast: number | null;
  maxGradNorm: number | null;
  lrSchedule: CosineLrSchedule | null;
  accum: number;
  uniformLoss: number | null;
  trainingPaused: boolean;
  /** Monotonic clock when **`phase`** became **`training`**; drives **`trainWallActiveSec`**. */
  trainingPhaseStartedAtMs: number | null;
  /** Completed training-pause intervals (ms), excluding **`trainingPauseBeganAtMs`** if currently paused. */
  trainingPausedAccumMs: number;
  /** Start of the current pause segment; **`null`** when not paused. */
  trainingPauseBeganAtMs: number | null;
  /** Set when the step loop finishes so **`trainWallActiveSec`** does not include checkpoint I/O. */
  trainingWallStoppedAtMs: number | null;
}

const SPARKLINE_MAX = 64;
const LOSS_SMA_WINDOW = 20;

/** Mean of the last **`LOSS_SMA_WINDOW`** finite values in **`losses`**. */
function trailingSma(losses: readonly number[], window: number): number | null {
  if (losses.length === 0) return null;
  const start = Math.max(0, losses.length - window);
  let sum = 0;
  let count = 0;
  for (let i = start; i < losses.length; i++) {
    const v = losses[i];
    if (v === undefined || !Number.isFinite(v)) continue;
    sum += v;
    count += 1;
  }
  return count === 0 ? null : sum / count;
}

const LN2 = Math.log(2);

export function createInitialTrainDashboardDraft(opts: {
  readonly outDirDisplay: string;
  readonly configTag: string;
  readonly seqLen: number;
  readonly totalTrainSteps: number;
}): TrainDashboardDraft {
  const t = Date.now();
  return {
    phase: 'starting',
    detailLine: '',
    startedAtMs: t,
    outDirDisplay: opts.outDirDisplay,
    configTag: opts.configTag,
    archSummary: null,
    gpuAdapterSummary: null,
    gpuSoftwareWarning: null,
    fileProgress: null,
    encodeProgress: null,
    saveProgress: null,
    corpusChars: null,
    corpusTokens: null,
    seqLen: opts.seqLen,
    totalTrainSteps: opts.totalTrainSteps,
    trainStepCompleted: 0,
    trainLossLast: 0,
    trainLrDisplay: 0,
    tokensPerSecApprox: 0,
    recentLosses: [],
    lossHistory: new MetricSeries(),
    gradNormHistory: new MetricSeries(),
    valHistory: [],
    gradNormLast: null,
    maxGradNorm: null,
    lrSchedule: null,
    accum: 1,
    uniformLoss: null,
    trainingPaused: false,
    trainingPhaseStartedAtMs: null,
    trainingPausedAccumMs: 0,
    trainingPauseBeganAtMs: null,
    trainingWallStoppedAtMs: null,
  };
}

export function finalizeTrainDashboardDraft(draft: TrainDashboardDraft): TrainDashboardSnapshot {
  const start = draft.startedAtMs;
  const sliceFrom = Math.max(0, draft.recentLosses.length - SPARKLINE_MAX);
  const now = Date.now();
  const tTrain0 = draft.trainingPhaseStartedAtMs;
  let trainWallActiveSec = 0;
  if (tTrain0 !== null) {
    const stoppedAt = draft.trainingWallStoppedAtMs;
    if (stoppedAt !== null) {
      trainWallActiveSec = Math.max(0, (stoppedAt - tTrain0 - draft.trainingPausedAccumMs) / 1000);
    } else {
      const pauseTailMs =
        draft.trainingPauseBeganAtMs !== null ? now - draft.trainingPauseBeganAtMs : 0;
      trainWallActiveSec = Math.max(
        0,
        (now - tTrain0 - draft.trainingPausedAccumMs - pauseTailMs) / 1000,
      );
    }
  }
  const sma20 = trailingSma(draft.recentLosses, LOSS_SMA_WINDOW);
  const meanCharsPerToken =
    draft.corpusChars !== null && draft.corpusTokens !== null && draft.corpusTokens > 0
      ? draft.corpusChars / draft.corpusTokens
      : null;
  const bitsPerCharSma20 =
    sma20 !== null && meanCharsPerToken !== null && meanCharsPerToken > 0
      ? sma20 / LN2 / meanCharsPerToken
      : null;
  const perplexitySma20 = sma20 !== null && Number.isFinite(sma20) ? Math.exp(sma20) : null;

  return {
    phase: draft.phase,
    detailLine: draft.detailLine,
    startedAtMs: start,
    nowElapsedSec: (now - start) / 1000,
    trainWallActiveSec,
    outDirDisplay: draft.outDirDisplay,
    configTag: draft.configTag,
    archSummary: draft.archSummary,
    gpuAdapterSummary: draft.gpuAdapterSummary,
    gpuSoftwareWarning: draft.gpuSoftwareWarning,
    fileProgress: draft.fileProgress,
    encodeProgress: draft.encodeProgress,
    saveProgress: draft.saveProgress,
    corpusChars: draft.corpusChars,
    corpusTokens: draft.corpusTokens,
    seqLen: draft.seqLen,
    totalTrainSteps: draft.totalTrainSteps,
    trainStepCompleted: draft.trainStepCompleted,
    trainLossLast: draft.trainLossLast,
    trainLossSma20: sma20,
    trainLrDisplay: draft.trainLrDisplay,
    tokensPerSecApprox: draft.tokensPerSecApprox,
    recentLosses: draft.recentLosses.slice(sliceFrom),
    meanCharsPerToken,
    bitsPerCharSma20,
    perplexitySma20,
    lossHistory: draft.lossHistory.snapshot(),
    gradNormHistory: draft.gradNormHistory.snapshot(),
    valHistory: draft.valHistory.slice(),
    valLossLast: draft.valHistory[draft.valHistory.length - 1]?.loss ?? null,
    gradNormLast: draft.gradNormLast,
    maxGradNorm: draft.maxGradNorm,
    lrSchedule: draft.lrSchedule,
    accum: draft.accum,
    uniformLoss: draft.uniformLoss,
    trainingPaused: draft.trainingPaused,
  };
}

/** Coalesced **`paint`** cadence while bursty **`schedule`** callers run (merge loop, checkpoints). */
export function createTrainDashboardThrottle(
  minIntervalMs: number,
  paint: () => void,
): { readonly schedule: () => void; readonly flush: () => void } {
  let lastPaintAtMs = 0;
  let timeout: ReturnType<typeof globalThis.setTimeout> | null = null;

  function doPaint(): void {
    timeout = null;
    paint();
    lastPaintAtMs = Date.now();
  }

  return {
    schedule() {
      if (timeout !== null) {
        return;
      }
      const now = Date.now();
      if (lastPaintAtMs === 0 || now - lastPaintAtMs >= minIntervalMs) {
        doPaint();
      } else {
        timeout = globalThis.setTimeout(doPaint, minIntervalMs - (now - lastPaintAtMs));
      }
    },

    flush() {
      if (timeout !== null) {
        globalThis.clearTimeout(timeout);
        timeout = null;
      }
      doPaint();
    },
  };
}
