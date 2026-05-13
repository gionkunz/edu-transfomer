import {
  readPackedCheckpointFiles,
  writePackedCheckpointFiles,
  type PackedCheckpointSidecar,
  type TensorPayload,
} from '@llm-transformer-diy/io';
import {
  assertValidGptConfig,
  collectGptParameters,
  createGptModel,
  gptParameterTensorNames,
  type GptConfig,
  type GptModel,
} from '@llm-transformer-diy/model';
import {
  copyF32Elems,
  createFromArray,
  destroy,
  productOfShape,
  toArray,
  type TensorRuntime,
} from '@llm-transformer-diy/tensor';

export type SaveGptCheckpointOptions = {
  readonly rt: TensorRuntime;
  readonly model: GptModel;
  readonly binPath: string;
  readonly jsonPath: string;
  /** Stored verbatim as **`sidecar.config`** (e.g. `GptConfig`). */
  readonly config: unknown;
  readonly trainingStep?: number;
  /** Fires after **`toArray`** for each exported parameter (CPU readback telemetry). */
  readonly onCheckpointTensor?: (info: {
    readonly index: number;
    readonly name: string;
    readonly total: number;
  }) => void;
};

/**
 * Writes **f32** weights for every **`collectGptParameters`** tensor (CPU readback per tensor).
 */
export async function saveGptCheckpoint(opts: SaveGptCheckpointOptions): Promise<void> {
  const params = collectGptParameters(opts.model);
  const names = gptParameterTensorNames(opts.model);
  if (names.length !== params.length) {
    throw new Error(
      `saveGptCheckpoint: name/param mismatch (${String(names.length)} vs ${String(params.length)})`,
    );
  }

  const tensors: Record<string, TensorPayload> = {};
  for (let i = 0; i < params.length; i++) {
    const name = names[i];
    const p = params[i];
    if (name === undefined || p === undefined) {
      throw new Error('saveGptCheckpoint: internal index error');
    }
    const data = await toArray(opts.rt, p.tensor);
    tensors[name] = { shape: [...p.tensor.shape], data: new Float32Array(data) };
    opts.onCheckpointTensor?.({ index: i, name, total: names.length });
  }

  await writePackedCheckpointFiles({
    binPath: opts.binPath,
    jsonPath: opts.jsonPath,
    config: opts.config,
    tensors,
    ...(opts.trainingStep !== undefined ? { metadata: { step: opts.trainingStep } } : {}),
  });
}

export async function loadGptCheckpointTensors(
  binPath: string,
  jsonPath: string,
): Promise<{
  readonly sidecar: PackedCheckpointSidecar;
  readonly tensors: Record<string, Float32Array>;
}> {
  return readPackedCheckpointFiles(binPath, jsonPath);
}

/**
 * Overwrites **`model`** parameter GPU buffers with checkpoint **`Float32`** payloads (**names** must match
 * {@link gptParameterTensorNames}).
 */
export function hydrateGptModelWeights(
  rt: TensorRuntime,
  model: GptModel,
  tensors: Readonly<Record<string, Float32Array>>,
): void {
  const names = gptParameterTensorNames(model);
  const params = collectGptParameters(model);
  if (names.length !== params.length) {
    throw new Error(
      `hydrateGptModelWeights: names/params length mismatch (${String(names.length)} vs ${String(params.length)})`,
    );
  }
  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    const p = params[i];
    const data = name !== undefined ? tensors[name] : undefined;
    if (name === undefined || p === undefined) {
      throw new Error('hydrateGptModelWeights: internal index error');
    }
    if (!data) {
      throw new Error(`hydrateGptModelWeights: checkpoint missing tensor "${name}"`);
    }
    const t = p.tensor;
    const n = productOfShape(t.shape);
    if (data.length !== n) {
      throw new Error(
        `hydrateGptModelWeights: shape mismatch for "${name}" (expected ${String(n)} elements, got ${String(data.length)})`,
      );
    }
    const src = createFromArray(rt, data, [...t.shape]);
    try {
      copyF32Elems(rt, src, 0, t, 0, n);
    } finally {
      destroy(src);
    }
  }
}

function parseGptConfig(raw: unknown): GptConfig {
  if (raw === null || typeof raw !== 'object') {
    throw new Error('checkpoint: config must be an object (GptConfig)');
  }
  const o = raw as Record<string, unknown>;
  const config: GptConfig = {
    vocabSize: Number(o.vocabSize),
    maxSeqLen: Number(o.maxSeqLen),
    dModel: Number(o.dModel),
    nHeads: Number(o.nHeads),
    dFf: Number(o.dFf),
    nLayers: Number(o.nLayers),
  };
  assertValidGptConfig(config);
  return config;
}

/**
 * Builds a **`GptModel`** from disk weights (**bin** + **JSON** sidecar).
 */
export async function loadGptModelFromCheckpoint(
  rt: TensorRuntime,
  binPath: string,
  jsonPath: string,
): Promise<{ readonly model: GptModel; readonly config: GptConfig }> {
  const { sidecar, tensors } = await loadGptCheckpointTensors(binPath, jsonPath);
  const config = parseGptConfig(sidecar.config);
  const model = createGptModel(rt, config);
  hydrateGptModelWeights(rt, model, tensors);
  return { model, config };
}
