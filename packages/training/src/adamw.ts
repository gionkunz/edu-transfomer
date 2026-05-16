import type { Parameter, ParameterRegistry } from '@llm-transformer-diy/autograd';
import type { TensorRuntime, Tensor } from '@llm-transformer-diy/tensor';
import {
  adamWStepInPlace,
  cast,
  copyContiguousTensorBytes,
  createZeros,
  destroy,
  floatDtypeOf,
  type AdamWStepHyperparams,
} from '@llm-transformer-diy/tensor';

/** Default β₁ (matches `docs/project.md`). */
export const ADAMW_DEFAULT_BETA1 = 0.9;

/** Default β₂ (matches `docs/project.md`). */
export const ADAMW_DEFAULT_BETA2 = 0.95;

/** Default ε. */
export const ADAMW_DEFAULT_EPS = 1e-8;

/** Default weight-decay λ before per-parameter scaling. */
export const ADAMW_DEFAULT_WEIGHT_DECAY = 0.1;

export type AdamWHyperparams = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
};

export type AdamWOptions = Partial<AdamWHyperparams> & {
  /**
   * Multiplier on `weightDecay` per parameter (`0` skips decay for biases, norms,
   * embeddings, … — see `docs/project.md`).
   */
  weightDecayScale?: (param: Parameter) => number;
};

function mergeDefaults(partial: AdamWOptions | undefined): AdamWHyperparams {
  return {
    lr: partial?.lr ?? 3e-4,
    beta1: partial?.beta1 ?? ADAMW_DEFAULT_BETA1,
    beta2: partial?.beta2 ?? ADAMW_DEFAULT_BETA2,
    eps: partial?.eps ?? ADAMW_DEFAULT_EPS,
    weightDecay: partial?.weightDecay ?? ADAMW_DEFAULT_WEIGHT_DECAY,
  };
}

/**
 * AdamW with fused GPU update; owns per-parameter first/second moment tensors.
 */
export class AdamW {
  private readonly rt: TensorRuntime;
  private readonly reg: ParameterRegistry;
  private readonly m = new Map<Parameter, Tensor>();
  private readonly v = new Map<Parameter, Tensor>();
  private t = 0;
  private hyper: AdamWHyperparams;
  private readonly weightDecayScale: (param: Parameter) => number;

  constructor(rt: TensorRuntime, reg: ParameterRegistry, options?: AdamWOptions) {
    this.rt = rt;
    this.reg = reg;
    this.hyper = mergeDefaults(options);
    this.weightDecayScale = options?.weightDecayScale ?? (() => 1);
  }

  setLr(lr: number): void {
    this.hyper = { ...this.hyper, lr };
  }

  /** Increments the timestep and applies AdamW to every parameter whose `grad` is set. */
  step(): void {
    this.t += 1;
    const biasCorrection1 = 1 - Math.pow(this.hyper.beta1, this.t);
    const biasCorrection2 = 1 - Math.pow(this.hyper.beta2, this.t);

    const hyperKernel: AdamWStepHyperparams = {
      lr: this.hyper.lr,
      beta1: this.hyper.beta1,
      beta2: this.hyper.beta2,
      eps: this.hyper.eps,
      weightDecay: 0,
      biasCorrection1,
      biasCorrection2,
    };

    for (const p of this.reg.params) {
      if (!p.grad) {
        continue;
      }
      let mT = this.m.get(p);
      let vT = this.v.get(p);
      if (!mT || !vT) {
        mT = createZeros(this.rt, p.tensor.shape, 'f32');
        vT = createZeros(this.rt, p.tensor.shape, 'f32');
        this.m.set(p, mT);
        this.v.set(p, vT);
      }
      hyperKernel.weightDecay = this.hyper.weightDecay * this.weightDecayScale(p);
      if (p.masterTensor === null) {
        adamWStepInPlace(this.rt, p.tensor, p.grad, mT, vT, hyperKernel);
      } else {
        const gradF32 = cast(this.rt, p.grad, 'f32');
        try {
          adamWStepInPlace(this.rt, p.masterTensor, gradF32, mT, vT, hyperKernel);
        } finally {
          destroy(gradF32);
        }
        const castDown = cast(this.rt, p.masterTensor, floatDtypeOf(p.tensor));
        try {
          copyContiguousTensorBytes(this.rt, castDown, p.tensor);
        } finally {
          destroy(castDown);
        }
      }
      destroy(p.grad);
      p.grad = null;
    }
  }

  /** Frees moment buffers; call when the trainer no longer needs this optimizer. */
  dispose(): void {
    for (const t of this.m.values()) {
      destroy(t);
    }
    for (const t of this.v.values()) {
      destroy(t);
    }
    this.m.clear();
    this.v.clear();
  }
}
