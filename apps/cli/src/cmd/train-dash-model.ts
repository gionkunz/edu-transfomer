import type { TrainDashboardSnapshot, TrainPipelinePhase } from './train-dash-types.js';

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
}

const SPARKLINE_MAX = 64;

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
  };
}

export function finalizeTrainDashboardDraft(draft: TrainDashboardDraft): TrainDashboardSnapshot {
  const start = draft.startedAtMs;
  const sliceFrom = Math.max(0, draft.recentLosses.length - SPARKLINE_MAX);
  return {
    phase: draft.phase,
    detailLine: draft.detailLine,
    startedAtMs: start,
    nowElapsedSec: (Date.now() - start) / 1000,
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
    trainLrDisplay: draft.trainLrDisplay,
    tokensPerSecApprox: draft.tokensPerSecApprox,
    recentLosses: draft.recentLosses.slice(sliceFrom),
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
