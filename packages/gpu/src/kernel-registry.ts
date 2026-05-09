/** Maps stable kernel IDs to compiled {@link GPUComputePipeline}s. */
export class KernelRegistry {
  readonly #pipelines = new Map<string, GPUComputePipeline>();

  register(id: string, pipeline: GPUComputePipeline): void {
    if (this.#pipelines.has(id)) {
      throw new Error(`KernelRegistry: duplicate kernel id "${id}"`);
    }
    this.#pipelines.set(id, pipeline);
  }

  get(id: string): GPUComputePipeline {
    const pipeline = this.#pipelines.get(id);
    if (!pipeline) {
      throw new Error(`KernelRegistry: unknown kernel id "${id}"`);
    }
    return pipeline;
  }
}
