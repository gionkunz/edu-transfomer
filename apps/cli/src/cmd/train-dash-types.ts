/** Phase machine for **`llm train`** dashboards (immutable snapshot consumed by wolf-tui). */

export type TrainPipelinePhase =
  | 'starting'
  | 'gpu_init'
  | 'tokenizer_load'
  | 'resolve_config'
  | 'discover_files'
  | 'read_corpus'
  | 'encode_corpus'
  | 'assemble_corpus'
  | 'verify_corpus'
  | 'build_model'
  | 'trainer_init'
  | 'training'
  | 'save_checkpoint'
  | 'finalize'
  | 'done';

export interface EncodeProgressSnapshot {
  readonly textChars: number;
  readonly byteLength: number;
  readonly mergeIteration: number;
  readonly mergeSubphase: 'expand' | 'merge';
  readonly tokenCount: number;
  /** Post-expand primitive token length **`N`**; used for worst-case merge bound **`N − 1`**. */
  readonly initialTokenCount: number;
  /** Wall seconds since corpus **`encodeAsync`** began (expand + merge). */
  readonly encodeElapsedSec: number;
  /** Worst-case maximum merge iterations: **`max(1, initialTokenCount − 1)`**. */
  readonly pessimisticMergeUpperBound: number;
  readonly pessimisticMergeRemaining: number;
  /** Smoothed merge throughput (merge-loop iterations/sec), **`null`** until estimated. */
  readonly mergesPerSec: number | null;
  /** Remaining pessimistic merges ÷ **`mergesPerSec`** when rate is trusted. */
  readonly pessimisticEtaSec: number | null;
  /** **`0`** in expand; **`0–100`** merge bar vs pessimistic upper bound (may finish sooner). */
  readonly mergeProgressPct: number;
}

export interface FileProgressSnapshot {
  readonly currentIndex: number;
  readonly totalFiles: number;
  readonly pathRel: string;
  readonly cumulativeChars: number;
}

export interface SaveProgressSnapshot {
  readonly tensorIndex: number;
  readonly totalTensors: number;
  readonly tensorName: string;
}

export interface TrainDashboardSnapshot {
  readonly phase: TrainPipelinePhase;
  readonly detailLine: string;
  readonly startedAtMs: number;
  readonly nowElapsedSec: number;
  readonly outDirDisplay: string;
  readonly configTag: string;
  readonly archSummary: string | null;
  /** Set after **`GpuContext.create`**; WebGPU **`GPUAdapter.info`** summary. */
  readonly gpuAdapterSummary: string | null;
  /** Non-null when WebGPU is probably LLVMpipe / SwiftShader (CPU). */
  readonly gpuSoftwareWarning: string | null;

  readonly fileProgress: FileProgressSnapshot | null;
  readonly encodeProgress: EncodeProgressSnapshot | null;
  readonly saveProgress: SaveProgressSnapshot | null;

  /** After encoding / assembly succeeds. */
  readonly corpusChars: number | null;
  readonly corpusTokens: number | null;
  readonly seqLen: number;
  readonly totalTrainSteps: number;

  readonly trainStepCompleted: number;
  readonly trainLossLast: number;
  readonly trainLrDisplay: number;
  readonly tokensPerSecApprox: number;
  readonly recentLosses: readonly number[];
}

/** Relative weights (**setup** funnel). Explicitly totals **100** for a clean macro bar. */
export const TRAIN_SETUP_PHASE_WEIGHTS: Readonly<Partial<Record<TrainPipelinePhase, number>>> = {
  starting: 4,
  gpu_init: 12,
  tokenizer_load: 6,
  resolve_config: 4,
  discover_files: 5,
  read_corpus: 9,
  encode_corpus: 36,
  assemble_corpus: 3,
  verify_corpus: 3,
  build_model: 12,
  trainer_init: 6,
};

/** Ordered funnel phases used by setup **`ProgressBar`** and stepper UX. */
export const TRAIN_SETUP_PIPELINE_ORDER: readonly TrainPipelinePhase[] = [
  'starting',
  'gpu_init',
  'tokenizer_load',
  'resolve_config',
  'discover_files',
  'read_corpus',
  'encode_corpus',
  'assemble_corpus',
  'verify_corpus',
  'build_model',
  'trainer_init',
];

export const SETUP_WEIGHT_SUM_TOTAL: number = TRAIN_SETUP_PIPELINE_ORDER.reduce(
  (sum, ph) => sum + (TRAIN_SETUP_PHASE_WEIGHTS[ph] ?? 0),
  0,
);

/** Cumulative normalized **0–100** through the current setup phase. */
export function setupMacroPct(phase: TrainPipelinePhase): number {
  if (
    phase === 'training' ||
    phase === 'save_checkpoint' ||
    phase === 'finalize' ||
    phase === 'done'
  ) {
    return 100;
  }
  let acc = 0;
  for (const p of TRAIN_SETUP_PIPELINE_ORDER) {
    acc += TRAIN_SETUP_PHASE_WEIGHTS[p] ?? 0;
    if (phase === p) {
      return Math.min(100, (acc / SETUP_WEIGHT_SUM_TOTAL) * 100);
    }
  }
  return Math.min(100, (acc / SETUP_WEIGHT_SUM_TOTAL) * 100);
}
