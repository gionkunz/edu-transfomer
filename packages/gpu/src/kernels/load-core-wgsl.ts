import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));

/**
 * Absolute path of a kernel source file.
 *
 * Two layouts exist: running from source (tests), the `.wgsl` files sit next to
 * this module; in the bundled CLI this module is inlined into `main.js` and the
 * build copies the shaders into a `kernels/` folder beside it.
 */
export function resolveWgslPath(file: string): string {
  const beside = path.join(MODULE_DIR, file);
  if (existsSync(beside)) {
    return beside;
  }
  const bundled = path.join(MODULE_DIR, 'kernels', file);
  if (existsSync(bundled)) {
    return bundled;
  }
  throw new Error(`WGSL kernel source not found: ${file} (looked in ${beside} and ${bundled})`);
}

function loadWgsl(relativePath: string): string {
  const file = relativePath.replace(/^\.\//, '');
  return rewriteMainForLargeDispatch(readFileSync(resolveWgslPath(file), 'utf8'));
}

const MAIN_HEADER_PATTERN =
  /fn\s+main\s*\(\s*@builtin\(global_invocation_id\)\s+gid:\s*vec3u\s*\)\s*\{/;

// WebGPU caps `dispatchWorkgroups` at 65535 per dimension. Once a kernel needs more
// workgroups than that on its single dimension (e.g. logits of `[batch*seq, vocab]`
// at vocab=30k, seq=1024), the host splits the dispatch across the Y dimension. The
// kernels still index work with a flat `gid.x`, so rewrite their `main` signature to
// derive the flat thread id from `workgroup_id`/`num_workgroups`/`local_invocation_id`
// instead of relying on the single-dim `global_invocation_id.x`.
export function rewriteMainForLargeDispatch(code: string): string {
  return code.replace(
    MAIN_HEADER_PATTERN,
    [
      'fn main(',
      '  @builtin(workgroup_id) _wg_id: vec3u,',
      '  @builtin(num_workgroups) _num_wg: vec3u,',
      '  @builtin(local_invocation_id) _l_id: vec3u,',
      ') {',
      '  let gid = vec3u((_wg_id.y * _num_wg.x + _wg_id.x) * 256u + _l_id.x, 0u, 0u);',
    ].join('\n'),
  );
}

function cat(...parts: string[]): string {
  return parts.join('\n');
}

const sharedBf16 = loadWgsl('./shared-bf16.wgsl');

export function loadMatMulWgsl(): string {
  return loadWgsl('./matmul.wgsl');
}
export function loadMatMulWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./matmul.bf16.frag.wgsl'));
}

export function loadTransposeWgsl(): string {
  return loadWgsl('./transpose.wgsl');
}
export function loadTransposeWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./transpose.bf16.frag.wgsl'));
}

export function loadElementwiseAddWgsl(): string {
  return loadWgsl('./elementwise-add.wgsl');
}
export function loadElementwiseAddWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./elementwise-add.bf16.frag.wgsl'));
}

export function loadElementwiseMulWgsl(): string {
  return loadWgsl('./elementwise-mul.wgsl');
}
export function loadElementwiseMulWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./elementwise-mul.bf16.frag.wgsl'));
}

export function loadElementwiseScaleWgsl(): string {
  return loadWgsl('./elementwise-scale.wgsl');
}
export function loadElementwiseScaleWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./elementwise-scale.bf16.frag.wgsl'));
}

export function loadElementwiseGeluWgsl(): string {
  return loadWgsl('./elementwise-gelu.wgsl');
}
export function loadElementwiseGeluWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./elementwise-gelu.bf16.frag.wgsl'));
}

export function loadElementwiseGeluBackwardWgsl(): string {
  return loadWgsl('./elementwise-gelu-backward.wgsl');
}
export function loadElementwiseGeluBackwardWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./elementwise-gelu-backward.bf16.frag.wgsl'));
}

export function loadReduceSumWgsl(): string {
  return loadWgsl('./reduce-sum.wgsl');
}
export function loadReduceSumWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./reduce-sum.bf16.frag.wgsl'));
}

export function loadReduceMaxWgsl(): string {
  return loadWgsl('./reduce-max.wgsl');
}
export function loadReduceMaxWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./reduce-max.bf16.frag.wgsl'));
}

export function loadSoftmaxWgsl(): string {
  return loadWgsl('./softmax.wgsl');
}
export function loadSoftmaxWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./softmax.bf16.frag.wgsl'));
}

export function loadSoftmaxBackwardWgsl(): string {
  return loadWgsl('./softmax-backward.wgsl');
}
export function loadSoftmaxBackwardWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./softmax-backward.bf16.frag.wgsl'));
}

export function loadRmsNormWgsl(): string {
  return loadWgsl('./rmsnorm.wgsl');
}
export function loadRmsNormWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./rmsnorm.bf16.frag.wgsl'));
}

export function loadRmsNormGradInputWgsl(): string {
  return loadWgsl('./rmsnorm-grad-input.wgsl');
}
export function loadRmsNormGradInputWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./rmsnorm-grad-input.bf16.frag.wgsl'));
}
export function loadRmsNormGradGammaWgsl(): string {
  return loadWgsl('./rmsnorm-grad-gamma.wgsl');
}
export function loadRmsNormGradGammaWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./rmsnorm-grad-gamma.bf16.frag.wgsl'));
}

export function loadEmbeddingGatherWgsl(): string {
  return loadWgsl('./embedding-gather.wgsl');
}
export function loadEmbeddingGatherWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./embedding-gather.bf16.frag.wgsl'));
}

export function loadEmbeddingScatterAccumWgsl(): string {
  return loadWgsl('./embedding-scatter-accum.wgsl');
}
export function loadEmbeddingScatterAccumWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./embedding-scatter-accum.bf16.frag.wgsl'));
}

export function loadCrossEntropyFromLogitsWgsl(): string {
  return loadWgsl('./cross-entropy-from-logits.wgsl');
}
export function loadCrossEntropyFromLogitsWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./cross-entropy-from-logits.bf16.frag.wgsl'));
}

export function loadCrossEntropyLogitsGradWgsl(): string {
  return loadWgsl('./cross-entropy-logits-grad.wgsl');
}
export function loadCrossEntropyLogitsGradWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./cross-entropy-logits-grad.bf16.frag.wgsl'));
}

export function loadBroadcastFillScalarWgsl(): string {
  return loadWgsl('./broadcast-fill-scalar.wgsl');
}
export function loadBroadcastFillScalarWgslBf16(): string {
  return cat(sharedBf16, loadWgsl('./broadcast-fill-scalar.bf16.frag.wgsl'));
}

export function loadCastF32ToBf16Wgsl(): string {
  return cat(sharedBf16, loadWgsl('./cast-f32-to-bf16.frag.wgsl'));
}
export function loadCastBf16ToF32Wgsl(): string {
  return cat(sharedBf16, loadWgsl('./cast-bf16-to-f32.frag.wgsl'));
}

export function loadAdamWStepWgsl(): string {
  return loadWgsl('./adamw-step.wgsl');
}
