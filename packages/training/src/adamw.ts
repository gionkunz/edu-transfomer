import type { Parameter, ParameterRegistry } from '@llm-transformer-diy/autograd';
import type { TensorRuntime, Tensor } from '@llm-transformer-diy/tensor';
import {
  adamWStepInPlace,
  cast,
  copyContiguousTensorBytes,
  createFromArray,
  createZeros,
  destroy,
  floatDtypeOf,
  productOfShape,
  toArray,
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

  /** Optimizer step counter **`t`** (Adam bias-correction exponent). */
  getTimestep(): number {
    return this.t;
  }

  /** Restore **`t`** when resuming from a checkpoint. */
  setTimestep(t: number): void {
    if (!Number.isInteger(t) || t < 0) {
      throw new Error(`AdamW.setTimestep: t must be a non-negative integer (got ${String(t)})`);
    }
    this.t = t;
  }

  /**
   * Serializes first/second moment buffers keyed by stable parameter names.
   * Parameters without allocated moments yet export zero-filled **`m`/`v`** arrays.
   */
  async exportNamedMoments(
    rt: TensorRuntime,
    paramNames: readonly string[],
  ): Promise<Record<string, { readonly m: Float32Array; readonly v: Float32Array }>> {
    if (paramNames.length !== this.reg.params.length) {
      throw new Error(
        `AdamW.exportNamedMoments: expected ${String(this.reg.params.length)} names, got ${String(paramNames.length)}`,
      );
    }
    const out: Record<string, { m: Float32Array; v: Float32Array }> = {};
    for (let i = 0; i < this.reg.params.length; i++) {
      const name = paramNames[i];
      const p = this.reg.params[i];
      if (name === undefined || p === undefined) {
        throw new Error('AdamW.exportNamedMoments: internal index error');
      }
      const n = productOfShape(p.tensor.shape);
      let mT = this.m.get(p);
      let vT = this.v.get(p);
      if (!mT || !vT) {
        out[name] = { m: new Float32Array(n), v: new Float32Array(n) };
        continue;
      }
      const mHost = await toArray(rt, mT);
      const vHost = await toArray(rt, vT);
      out[name] = { m: new Float32Array(mHost), v: new Float32Array(vHost) };
    }
    return out;
  }

  /**
   * Loads moment buffers from a checkpoint. Replaces any existing moment tensors for each parameter.
   */
  hydrateNamedMoments(
    rt: TensorRuntime,
    paramNames: readonly string[],
    moments: Readonly<Record<string, { readonly m: Float32Array; readonly v: Float32Array }>>,
  ): void {
    if (paramNames.length !== this.reg.params.length) {
      throw new Error(
        `AdamW.hydrateNamedMoments: expected ${String(this.reg.params.length)} names, got ${String(paramNames.length)}`,
      );
    }
    for (let i = 0; i < this.reg.params.length; i++) {
      const name = paramNames[i];
      const p = this.reg.params[i];
      if (name === undefined || p === undefined) {
        throw new Error('AdamW.hydrateNamedMoments: internal index error');
      }
      const entry = moments[name];
      if (!entry) {
        throw new Error(`AdamW.hydrateNamedMoments: checkpoint missing moments for "${name}"`);
      }
      const n = productOfShape(p.tensor.shape);
      if (entry.m.length !== n || entry.v.length !== n) {
        throw new Error(
          `AdamW.hydrateNamedMoments: shape mismatch for "${name}" (expected ${String(n)} elements)`,
        );
      }
      const prevM = this.m.get(p);
      const prevV = this.v.get(p);
      if (prevM) {
        destroy(prevM);
      }
      if (prevV) {
        destroy(prevV);
      }
      const mT = createFromArray(rt, entry.m, [...p.tensor.shape], 'f32');
      const vT = createFromArray(rt, entry.v, [...p.tensor.shape], 'f32');
      this.m.set(p, mT);
      this.v.set(p, vT);
    }
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
