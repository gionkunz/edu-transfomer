import {
  backward,
  clipGradNorm,
  createLeafVariable,
  crossEntropyMeanFromLogits,
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
};

/**
 * Minimal single-process trainer: one forward (GPT + mean CE), backward, clip, AdamW.
 *
 * Does not yet reclaim intermediate graph buffers; keep **`steps`** modest or extend with teardown.
 */
export class Trainer {
  private readonly rt: TensorRuntime;
  private readonly model: GptModel;
  private readonly registry: ParameterRegistry;
  private readonly optimizer: AdamW;
  private readonly options: TrainerOptions;

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
      throw new Error('trainer: non-finite loss');
    }

    backward(this.rt, loss);

    if (Number.isFinite(maxGradNorm) && maxGradNorm > 0) {
      await clipGradNorm(this.rt, this.registry, maxGradNorm);
    }

    this.optimizer.step();

    onStep?.({ trainingStepIndex, loss: lossScalar, lr });

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
