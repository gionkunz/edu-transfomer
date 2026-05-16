import { releasePooledOrDestroy } from '@llm-transformer-diy/gpu';
import type { Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';
import {
  add,
  cast,
  createZeros,
  destroy,
  mul,
  reshape,
  scale,
  sumFlat,
  toArray,
} from '@llm-transformer-diy/tensor';

import { topologicalOrder } from './backward';

import type { Parameter, ParameterRegistry, Variable } from './types';

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
 * Sum of `p.tensor.length` across every registered parameter (compute weights only;
 * {@link Parameter.masterTensor} is excluded).
 */
export function totalParams(reg: ParameterRegistry): number {
  let total = 0;
  for (const p of reg.params) {
    total += p.tensor.length;
  }
  return total;
}

async function sumGradNormSq(rt: TensorRuntime, reg: ParameterRegistry): Promise<number> {
  let acc = createZeros(rt, [1]);
  try {
    for (const p of reg.params) {
      if (!p.grad) {
        continue;
      }
      const g = p.grad.shape.length === 1 ? p.grad : reshape(p.grad, [p.grad.length]);
      const gF32 = g.dtype === 'f32' ? g : cast(rt, g, 'f32');
      try {
        const sq = mul(rt, gF32, gF32);
        const part = sumFlat(rt, sq);
        const next = add(rt, acc, part);
        destroy(acc);
        destroy(sq);
        destroy(part);
        acc = next;
      } finally {
        if (gF32 !== g) {
          destroy(gF32);
        }
      }
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
  if (!Number.isFinite(sumSq) || !Number.isFinite(norm)) {
    throw new Error(
      'clipGradNorm: global gradient L2 norm is non-finite (NaN/Inf in some parameter gradient — refusing to scale, which would poison every grad tensor)',
    );
  }
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

/**
 * After {@link import('./backward').backward}, release GPU buffers for every graph
 * node except registered parameters and caller-owned leaves (`creator === null`
 * and not in the registry). Parameter `.tensor` / `.grad` are kept for
 * {@link clipGradNorm} / optimizers. Skipping duplicate {@link destroy} calls
 * when several variables share one buffer (e.g. {@link import('./ops').reshape}).
 */
export function disposeAutogradGraphExceptParameters(
  output: Variable,
  registry: ParameterRegistry,
): void {
  type BufferId = Tensor['buffer'];
  const paramSet = new Set<Variable>(registry.params);
  const protectedBuffers = new Set<BufferId>();

  for (const p of registry.params) {
    protectedBuffers.add(p.tensor.buffer);
    if (p.masterTensor !== null) {
      protectedBuffers.add(p.masterTensor.buffer);
    }
    if (p.grad) {
      protectedBuffers.add(p.grad.buffer);
    }
  }

  const order = topologicalOrder(output);
  for (const v of order) {
    if (v.creator === null && !paramSet.has(v)) {
      protectedBuffers.add(v.tensor.buffer);
    }
  }

  const destroyed = new Set<BufferId>();

  function releaseBuffer(buf: BufferId): void {
    if (destroyed.has(buf) || protectedBuffers.has(buf)) {
      return;
    }
    destroyed.add(buf);
    releasePooledOrDestroy(buf);
  }

  function releaseMaybeTensor(t: Tensor | null | undefined): void {
    if (t == null) {
      return;
    }
    releaseBuffer(t.buffer);
  }

  for (const v of order) {
    if (paramSet.has(v)) {
      continue;
    }
    releaseMaybeTensor(v.grad);
    releaseMaybeTensor(v.tensor);
  }
}
