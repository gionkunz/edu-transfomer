import { GpuContext, KernelRegistry, registerCoreKernels } from '@llm-transformer-diy/gpu';
import type { TensorRuntime } from '@llm-transformer-diy/tensor';

/** Creates a **TensorRuntime** with all **core** WGSL kernels registered. */
export async function createTensorRuntime(): Promise<TensorRuntime> {
  const ctx = await GpuContext.create();
  const registry = new KernelRegistry();
  registerCoreKernels(ctx.device, registry);
  return { ctx, registry };
}
