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
  /** Per-region outer-pass count for the region currently being encoded. */
  readonly mergeIteration: number;
  readonly mergeSubphase: 'expand' | 'merge';
  readonly tokenCount: number;
  /** Post-expand primitive token length **`N`** (corpus-wide, prior to merging). */
  readonly initialTokenCount: number;
  /** Wall seconds since corpus **`encodeUint32Async`** began (expand + merge). */
  readonly encodeElapsedSec: number;
  /** Active tokenizer pretokenizer policy (label). */
  readonly pretokenizerPolicy: 'word-and-punct-v1';
  /** Smoothed regions-per-second throughput, **`null`** until estimated. */
  readonly regionsPerSec: number | null;
  /** ETA based on `regionsRemaining * meanPassesPerRegion / passRate` when rates are trusted. */
  readonly etaSec: number | null;
  /** Zero-based index of the region currently being encoded. */
  readonly regionIndex: number;
  /** Total regions in the active encode call. */
  readonly regionsTotal: number;
  /** Running average of outer passes per completed region. */
  readonly meanPassesPerRegion: number | null;
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
  /** Wall seconds since **`startedAtMs`** (full run, including setup and paused training time). */
  readonly nowElapsedSec: number;
  /**
   * Seconds since the **`training`** phase began, excluding time spent with training paused (**`p`**).
   * After the step loop completes, this value freezes so checkpoint I/O is not counted.
   * **`0`** before the training phase; used for the training progress wall clock and ETA.
   */
  readonly trainWallActiveSec: number;
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
  /**
   * When **`true`**, the training loop yields between steps (no **`trainStep`** / batch prep)
   * so the GPU stays idle until resumed (**`p`** in the TUI).
   */
  readonly trainingPaused: boolean;
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
