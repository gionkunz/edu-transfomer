import {
  backward,
  clipGradNorm,
  createLeafVariable,
  crossEntropyMeanFromLogits,
  disposeAutogradGraphExceptParameters,
  registerParameter,
  type Parameter,
  type ParameterRegistry,
  zeroGrad,
} from '@llm-transformer-diy/autograd';
import { forwardGpt, type GptModel } from '@llm-transformer-diy/model';
import { toArray, type Tensor, type TensorRuntime } from '@llm-transformer-diy/tensor';

import type { AdamW } from './adamw';
import type { CosineLrSchedule } from './scheduler';
import { getCosineLr } from './scheduler';

export type TrainerStepInfo = {
  readonly trainingStepIndex: number;
  readonly loss: number;
  readonly lr: number;
};

export type TrainerOptions = {
  /** Global L2 gradient clip (`Infinity` disables numerically — prefer large number or skip clip in future). */
  readonly maxGradNorm: number;
  readonly schedule: CosineLrSchedule;
  /** Optional hook after **`optimizer.step`** (for CLIs / logging). */
  readonly onStep?: (info: TrainerStepInfo) => void;
  /**
   * If Node was started with `--expose-gc`, force a major GC every N training
   * steps. Smooths out the long sawtooth where V8 lazily collects stale
   * `GPUBuffer` wrappers, trading a small predictable pause for unpredictable
   * multi-second incremental-marking stalls.
   *
   * `0` / unset → never call gc() explicitly. Ignored when `global.gc` is
   * unavailable (Node without `--expose-gc`).
   */
  readonly forceGcEverySteps?: number;
  /**
   * Emit a one-line buffer-pool stats summary to stderr every N steps. `0` /
   * unset → never. Used to verify the pool's hit-rate climbs to ~100% after
   * the warmup window and that VRAM stays bounded.
   */
  readonly logBufferPoolEverySteps?: number;
};

/**
 * Returns Node's `global.gc` only if it has been exposed via `--expose-gc`.
 * Resolved once on first call; subsequent calls are a plain function check.
 */
function resolveExposedGc(): (() => void) | null {
  const g = globalThis as { gc?: unknown };
  return typeof g.gc === 'function' ? (g.gc as () => void) : null;
}

/**
 * Minimal single-process trainer: one forward (GPT + mean CE), backward, graph
 * disposal (intermediate buffers), clip, AdamW.
 */
export class Trainer {
  private readonly rt: TensorRuntime;
  private readonly model: GptModel;
  private readonly registry: ParameterRegistry;
  private readonly optimizer: AdamW;
  private readonly options: TrainerOptions;
  private readonly exposedGc: (() => void) | null;

  constructor(
    rt: TensorRuntime,
    model: GptModel,
    registry: ParameterRegistry,
    optimizer: AdamW,
    options: TrainerOptions,
  ) {
    this.rt = rt;
    this.model = model;
    this.registry = registry;
    this.optimizer = optimizer;
    this.options = options;
    this.exposedGc = resolveExposedGc();
  }

  /**
   * Runs **`trainingStepIndex`** through the LR schedule, executes one LM step, returns **scalar** mean loss.
   */
  async trainStep(
    tokenInput: Tensor,
    tokenTargets: Tensor,
    trainingStepIndex: number,
  ): Promise<number> {
    const { maxGradNorm, schedule, onStep } = this.options;

    zeroGrad(this.registry);
    const lr = getCosineLr(trainingStepIndex, schedule);
    this.optimizer.setLr(lr);

    const logits = forwardGpt(this.rt, this.model, tokenInput);
    const targetsVar = createLeafVariable(tokenTargets, false);
    const loss = crossEntropyMeanFromLogits(this.rt, logits, targetsVar);

    const lossHost = await toArray(this.rt, loss.tensor);
    const lossScalar = lossHost[0];
    if (lossScalar === undefined || !Number.isFinite(lossScalar)) {
      const kind =
        lossScalar === undefined
          ? 'missing'
          : Number.isNaN(lossScalar)
            ? 'NaN'
            : lossScalar === Number.POSITIVE_INFINITY
              ? '+Infinity'
              : lossScalar === Number.NEGATIVE_INFINITY
                ? '-Infinity'
                : 'non-finite';
      throw new Error(
        `trainer: non-finite loss at step ${String(trainingStepIndex)} (${kind}). ` +
          `Often: exploding activations (try lower lrMax, smaller maxSeqLen), bad token id ≥ vocab in CE ` +
          `(re-encode corpus / delete token cache), or run with tighter maxGradNorm.`,
      );
    }

    backward(this.rt, loss);
    disposeAutogradGraphExceptParameters(loss, this.registry);

    if (Number.isFinite(maxGradNorm) && maxGradNorm > 0) {
      await clipGradNorm(this.rt, this.registry, maxGradNorm);
    }

    this.optimizer.step();

    onStep?.({ trainingStepIndex, loss: lossScalar, lr });

    const gcEvery = this.options.forceGcEverySteps ?? 0;
    if (
      gcEvery > 0 &&
      this.exposedGc !== null &&
      trainingStepIndex > 0 &&
      trainingStepIndex % gcEvery === 0
    ) {
      this.exposedGc();
    }

    // Periodic buffer-pool diagnostics so we can verify the pool is doing
    // what we expect at runtime: hit-rate should approach ~100% after warmup,
    // and idleBytes should stabilize quickly. Emitting to stderr keeps the
    // dashboard TUI clean.
    const poolEvery = this.options.logBufferPoolEverySteps ?? 0;
    if (poolEvery > 0 && trainingStepIndex > 0 && trainingStepIndex % poolEvery === 0) {
      const s = this.rt.ctx.bufferPool.stats();
      const total = s.acquireHits + s.acquireMisses;
      const hitPct = total > 0 ? ((s.acquireHits / total) * 100).toFixed(1) : 'n/a';
      const idleMib = (s.idleBytes / (1024 * 1024)).toFixed(1);
      console.error(
        `trainer: pool step=${String(trainingStepIndex)} ` +
          `hit=${hitPct}% (${String(s.acquireHits)}/${String(total)}) ` +
          `idle=${String(s.idleBuffers)} buffers / ${idleMib} MiB ` +
          `in ${String(s.buckets)} buckets ` +
          `overflow=${String(s.overflowDestroys)}`,
      );
    }

    return lossScalar;
  }
}

/**
 * Register every parameter in **`params`** on **`registry`** (append-only).
 */
export function registerModelParameters(
  registry: ParameterRegistry,
  params: readonly Parameter[],
): void {
  for (const p of params) {
    registerParameter(registry, p);
  }
}
