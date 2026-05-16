import { supportsDtype } from '@llm-transformer-diy/gpu';
import type { GpuContext } from '@llm-transformer-diy/gpu';
import type { FloatDType } from '@llm-transformer-diy/tensor';

export function parseOptionalComputeDtype(raw: string | undefined, flagLabel: string): FloatDType {
  if (raw === undefined || raw === '') {
    return 'f32';
  }
  const v = raw.trim().toLowerCase();
  if (v === 'f32' || v === 'bf16' || v === 'fp16') {
    return v;
  }
  throw new Error(`${flagLabel}: expected f32, bf16, or fp16 (got ${JSON.stringify(raw)})`);
}

/**
 * After **`GpuContext.create`**, fail fast when the user requested a dtype the runtime cannot use.
 */
export function assertRuntimeSupportsComputeDtype(
  ctx: GpuContext,
  dtype: FloatDType,
  commandLabel: string,
): void {
  if (!supportsDtype(ctx, dtype)) {
    throw new Error(
      `${commandLabel}: WebGPU runtime does not support compute dtype '${dtype}' ` +
        `(adapter: ${ctx.adapterLabel}).`,
    );
  }
}
