import type { Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';
import {
  add,
  assertDtype,
  createZeros,
  destroy,
  mul,
  reshape,
  scale,
  sumFlat,
  toArray,
} from '@llm-transformer-diy/tensor';

import type { Parameter, ParameterRegistry } from './types';

/** Build a fresh, empty {@link ParameterRegistry}. */
export function createParameterRegistry(): ParameterRegistry {
  return { params: [] };
}

/**
 * Append `p` to `reg`. The registry does **not** deduplicate registrations and
 * does **not** support removal; callers are responsible for not double-
 * registering.
 */
export function registerParameter(reg: ParameterRegistry, p: Parameter): void {
  reg.params.push(p);
}

/**
 * Snapshot of the registered parameters in registration order.
 *
 * Returns a defensive copy rather than a `readonly` view so a caller
 * iterating the result while another piece of code calls
 * {@link registerParameter} cannot observe partial state. The cost is a small
 * array allocation per call; for a model the size of `tiny`/`small` from
 * `docs/project.md` this is negligible.
 */
export function parameters(reg: ParameterRegistry): readonly Parameter[] {
  return [...reg.params];
}

/**
 * Set every registered parameter's `.grad` to `null`.
 *
 * Together with {@link import('./variable').accumulateGrad} (which assigns
 * gradients during backward), {@link clipGradNorm} (which may replace `.grad`
 * tensors), and this routine (which sets `.grad` to `null`), every write path
 * to {@link import('./types').Variable.grad} lives in this package.
 */
export function zeroGrad(reg: ParameterRegistry): void {
  for (const p of reg.params) {
    p.grad = null;
  }
}

/**
 * Sum of `p.tensor.length` across every registered parameter.
 *
 * Single-dtype (`f32`) in this scope, so the count is element-uniform.
 */
export function totalParams(reg: ParameterRegistry): number {
  let total = 0;
  for (const p of reg.params) {
    total += p.tensor.length;
  }
  return total;
}

function asFlatF32(t: Tensor): Tensor {
  assertDtype(t, 'f32');
  if (t.shape.length === 1) {
    return t;
  }
  return reshape(t, [t.length]);
}

/**
 * Σ ‖g‖²₂ over every registered parameter that currently has a gradient.
 */
async function sumGradNormSq(rt: TensorRuntime, reg: ParameterRegistry): Promise<number> {
  let acc = createZeros(rt, [1]);
  try {
    for (const p of reg.params) {
      if (!p.grad) {
        continue;
      }
      const g = asFlatF32(p.grad);
      const sq = mul(rt, g, g);
      const part = sumFlat(rt, sq);
      const next = add(rt, acc, part);
      destroy(acc);
      acc = next;
    }
    const host = await toArray(rt, acc);
    const sumSq = host[0];
    if (sumSq === undefined) {
      throw new Error('sumGradNormSq: empty readback');
    }
    return sumSq;
  } finally {
    destroy(acc);
  }
}

/**
 * Scales all accumulated gradients so the total L2 norm does not exceed **`maxNorm`**.
 * Parameters with `grad === null` are skipped.
 */
export async function clipGradNorm(
  rt: TensorRuntime,
  reg: ParameterRegistry,
  maxNorm: number,
): Promise<void> {
  if (!(maxNorm > 0) || !Number.isFinite(maxNorm)) {
    throw new Error('clipGradNorm: maxNorm must be finite and > 0');
  }
  const sumSq = await sumGradNormSq(rt, reg);
  const norm = Math.sqrt(sumSq);
  if (norm === 0 || norm <= maxNorm) {
    return;
  }
  const s = maxNorm / norm;
  for (const p of reg.params) {
    if (!p.grad) {
      continue;
    }
    const scaled = scale(rt, p.grad, s);
    destroy(p.grad);
    p.grad = scaled;
  }
}
