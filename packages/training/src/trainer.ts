import {
  backward,
  clipGradNorm,
  createLeafVariable,
  crossEntropyMeanFromLogits,
  disposeAutogradGraphExceptParameters,
  gradNorm,
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
import {
  buildNonFiniteLossDiagnostics,
  type TrainStepBatchContext,
} from './train-failure-diagnostics';

export type TrainerStepInfo = {
  readonly trainingStepIndex: number;
  readonly loss: number;
  readonly lr: number;
};

/** One sequence of a training step: **input** ids, shifted **targets**, optional diagnostics context. */
export type TrainerMicroBatch = {
  readonly input: Tensor;
  readonly targets: Tensor;
  readonly batch?: TrainStepBatchContext;
};

export type TrainerStepResult = {
  /** Mean loss over the micro-batches of this step. */
  readonly loss: number;
  readonly lr: number;
  /** Global gradient L2 norm **before** clipping; `null` unless clipping or `measureGradNorm` is on. */
  readonly gradNorm: number | null;
};

export type TrainerOptions = {
  /** Global L2 gradient clip (`Infinity` disables numerically — prefer large number or skip clip in future). */
  readonly maxGradNorm: number;
  readonly schedule: CosineLrSchedule;
  /** Optional hook after **`optimizer.step`** (for CLIs / logging). */
  readonly onStep?: (info: TrainerStepInfo) => void;
  /**
   * Read back the global gradient norm even when clipping is disabled (one extra
   * GPU readback per step). With clipping enabled the norm is always available.
   */
  readonly measureGradNorm?: boolean;
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
    batch?: TrainStepBatchContext,
  ): Promise<number> {
    const result = await this.trainStepAccumulated(
      [{ input: tokenInput, targets: tokenTargets, ...(batch !== undefined ? { batch } : {}) }],
      trainingStepIndex,
    );
    return result.loss;
  }

  /**
   * One optimizer step over **`microBatches.length`** sequences (gradient accumulation).
   *
   * Each sequence gets its own forward + backward; the backward seed is `1 / n`, and
   * gradients are **not** zeroed in between, so parameters end up holding the gradient
   * of the mean loss — the same update a real batch of `n` sequences would produce, at
   * the memory cost of a single sequence. Then: clip once, AdamW once.
   */
  async trainStepAccumulated(
    microBatches: readonly TrainerMicroBatch[],
    trainingStepIndex: number,
  ): Promise<TrainerStepResult> {
    const { maxGradNorm, schedule, onStep } = this.options;
    const n = microBatches.length;
    if (n === 0) {
      throw new Error('trainer: trainStepAccumulated needs at least one micro-batch');
    }

    zeroGrad(this.registry);
    const lr = getCosineLr(trainingStepIndex, schedule);
    this.optimizer.setLr(lr);

    let lossSum = 0;
    for (const mb of microBatches) {
      const logits = forwardGpt(this.rt, this.model, mb.input);
      const targetsVar = createLeafVariable(mb.targets, false);
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
        const seqLen = mb.input.shape[0];
        if (seqLen === undefined) {
          throw new Error(
            `trainer: non-finite loss at step ${String(trainingStepIndex)} (${kind})`,
          );
        }
        const detail = await buildNonFiniteLossDiagnostics(this.rt, {
          trainingStepIndex,
          kind,
          lr,
          logits,
          tokenInput: mb.input,
          targets: mb.targets,
          batch: mb.batch,
          seqLen,
          model: this.model,
        });
        throw new Error(detail);
      }
      lossSum += lossScalar;

      backward(this.rt, loss, 1 / n);
      this.disposeGraph(loss, mb.input, mb.targets);
    }
    const lossScalar = lossSum / n;

    let gradNormValue: number | null = null;
    if (Number.isFinite(maxGradNorm) && maxGradNorm > 0) {
      gradNormValue = await clipGradNorm(this.rt, this.registry, maxGradNorm);
    } else if (this.options.measureGradNorm === true) {
      gradNormValue = await gradNorm(this.rt, this.registry);
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

    return { loss: lossScalar, lr, gradNorm: gradNormValue };
  }

  /**
   * Forward-only mean cross-entropy on one sequence: no backward, no parameter
   * update. Used for validation loss on held-out text.
   */
  async evalLoss(tokenInput: Tensor, tokenTargets: Tensor): Promise<number> {
    const logits = forwardGpt(this.rt, this.model, tokenInput);
    const targetsVar = createLeafVariable(tokenTargets, false);
    const loss = crossEntropyMeanFromLogits(this.rt, logits, targetsVar);
    try {
      const host = await toArray(this.rt, loss.tensor);
      const v = host[0];
      if (v === undefined) {
        throw new Error('trainer: evalLoss readback was empty');
      }
      return v;
    } finally {
      this.disposeGraph(loss, tokenInput, tokenTargets);
    }
  }

  /**
   * Release every intermediate buffer of one forward (+ backward) graph.
   *
   * Explicit external-leaves list: anything the caller / model owns and reuses
   * across training steps. Without this, `disposeAutograd…` would either leak the
   * per-step internally-allocated leaves (positional indices, causal mask in
   * `forwardGpt`) or destroy long-lived shared leaves like `model.headExpand`.
   * Passing the exact set keeps both invariants while letting internal pos / mask
   * buffers return to the pool.
   */
  private disposeGraph(
    loss: Parameters<typeof backward>[1],
    tokenInput: Tensor,
    tokenTargets: Tensor,
  ): void {
    const externalLeaves: Tensor[] = [tokenInput, tokenTargets];
    for (const he of this.model.headExpand) {
      externalLeaves.push(he.tensor);
    }
    disposeAutogradGraphExceptParameters(loss, this.registry, externalLeaves);
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
