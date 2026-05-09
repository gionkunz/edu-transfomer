export type StorageBufferBindingType = 'storage' | 'read-only-storage' | 'uniform';

export type ComputeBindingDesc = {
  binding: number;
  visibility: GPUShaderStageFlags;
  buffer: {
    type: StorageBufferBindingType;
  };
};

export type SimpleComputePipelineArgs = {
  code: string;
  entryPoint: string;
  bindings: readonly ComputeBindingDesc[];
};

export function createShaderModule(device: GPUDevice, code: string): GPUShaderModule {
  return device.createShaderModule({ code });
}

/**
 * One bind-group layout (group 0), one compute entry point, explicit buffer bindings.
 */
export function createSimpleComputePipeline(
  device: GPUDevice,
  args: SimpleComputePipelineArgs,
): { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout } {
  const bindGroupLayout = device.createBindGroupLayout({
    entries: args.bindings.map((b) => ({
      binding: b.binding,
      visibility: b.visibility,
      buffer: { type: b.buffer.type },
    })),
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const shaderModule = createShaderModule(device, args.code);
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: args.entryPoint,
    },
  });

  return { pipeline, bindGroupLayout };
}
