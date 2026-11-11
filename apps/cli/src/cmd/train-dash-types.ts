/** Phase machine for **`llm train`** dashboards (immutable snapshot consumed by wolf-tui). */

import type { CosineLrSchedule } from '@llm-transformer-diy/training';

import type { MetricBucket } from './metric-series';

/** One validation measurement: mean held-out loss after **`step`** completed optimizer steps. */
export interface ValLossPoint {
  readonly step: number;
  readonly loss: number;
}

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
  readonly pretokenizerPolicy: 'gpt2-v1';
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
  /**
   * Simple moving average of the last **20** training losses (or fewer, if not
   * enough steps have completed yet). **`null`** before the first step.
   * Smooths the per-step jitter; pair with **`trainLossLast`** for "current vs
   * trend" reading.
   */
  readonly trainLossSma20: number | null;
  readonly trainLrDisplay: number;
  readonly tokensPerSecApprox: number;
  readonly recentLosses: readonly number[];
  /**
   * Mean characters per token observed during corpus encoding (i.e.
   * **`corpusChars / corpusTokens`**). The tokenizer-invariant rescaling factor
   * used by **`bitsPerCharSma20`** and per-character perplexity. **`null`** until
   * encoding completes.
   */
  readonly meanCharsPerToken: number | null;
  /**
   * Bits per character implied by **`trainLossSma20`** (loss in nats divided by
   * **`ln 2`** divided by **`meanCharsPerToken`**). The single tokenizer- and
   * model-invariant comparable training-quality metric. **`null`** before the
   * first step or before encoding completes.
   */
  readonly bitsPerCharSma20: number | null;
  /**
   * Per-token perplexity implied by **`trainLossSma20`**: **`exp(loss)`**.
   * **`null`** before the first step.
   */
  readonly perplexitySma20: number | null;
  /** Whole-run train loss, downsampled to a fixed number of buckets (see **`MetricSeries`**). */
  readonly lossHistory: readonly MetricBucket[];
  /** Whole-run pre-clip gradient L2 norm; empty when the norm is not measured. */
  readonly gradNormHistory: readonly MetricBucket[];
  /** Held-out loss measurements (**`--val-data`**); empty without validation data. */
  readonly valHistory: readonly ValLossPoint[];
  readonly valLossLast: number | null;
  readonly gradNormLast: number | null;
  /** Clip threshold, or **`null`** when clipping is off. */
  readonly maxGradNorm: number | null;
  /** LR schedule of the run, for the schedule chart; **`null`** before the trainer exists. */
  readonly lrSchedule: CosineLrSchedule | null;
  /** Sequences per optimizer step (**`--accum`**). */
  readonly accum: number;
  /** **`ln(vocabSize)`**: the loss of a model that predicts every token with equal probability. */
  readonly uniformLoss: number | null;
  /** Most recent diagnostic lines; the full log is `train.log` in the output directory. */
  readonly logTail: readonly string[];
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
