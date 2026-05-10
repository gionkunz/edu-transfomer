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
 * Together with {@link import('./variable').accumulateGrad}, this is the only
 * function in `@llm-transformer-diy/autograd` that writes to `Variable.grad`.
 * `null` (rather than a zero tensor) is the explicit "no contribution yet"
 * state used by `accumulateGrad`'s direct-assignment fast path.
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
