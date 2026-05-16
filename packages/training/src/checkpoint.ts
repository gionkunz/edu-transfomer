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
  copyContiguousTensorBytes,
  createFromArray,
  destroy,
  floatDtypeOf,
  productOfShape,
  toArray,
  type FloatDType,
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
 * Writes weights for every **`collectGptParameters`** tensor at each parameter's compute **`dtype`**
 * (CPU readback per tensor, **Float32Array** logical values encoded with **`packBf16FromFloat32`** /
 * **`packFp16FromFloat32`** when needed).
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
    const dtype = floatDtypeOf(p.tensor);
    tensors[name] = {
      shape: [...p.tensor.shape],
      data: new Float32Array(data),
      ...(dtype !== 'f32' ? { dtype } : {}),
    };
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

function assertCheckpointCompatibleWithComputeDtype(
  sidecar: PackedCheckpointSidecar,
  computeDtype: FloatDType,
  allowDtypeUpcast: boolean | undefined,
): void {
  for (const [name, entry] of Object.entries(sidecar.tensors)) {
    const stored = entry.dtype;
    if (stored === computeDtype) {
      continue;
    }
    if (stored === 'f32' && computeDtype !== 'f32') {
      continue;
    }
    if (stored !== 'f32' && computeDtype === 'f32') {
      if (allowDtypeUpcast === true) {
        continue;
      }
      throw new Error(
        `loadGptModelFromCheckpoint: checkpoint tensor "${name}" is stored as '${stored}' ` +
          `but compute dtype is f32. Pass { allowDtypeUpcast: true } to decode 16-bit weights to f32, or use --dtype ${stored}.`,
      );
    }
    throw new Error(
      `loadGptModelFromCheckpoint: checkpoint tensor "${name}" is stored as '${stored}' ` +
        `which is incompatible with compute dtype '${computeDtype}'.`,
    );
  }
}

/**
 * Overwrites **`model`** parameter buffers. **`tensors`** values are logical **`f32`** host arrays
 * (**16-bit** checkpoints are already unpacked in **`readPackedCheckpointFiles`**); each value is
 * narrowed to **`parameter.tensor.dtype`** on upload (**`createFromArray`**).
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
    const dt = floatDtypeOf(t);
    const src = createFromArray(rt, data, [...t.shape], dt);
    try {
      copyContiguousTensorBytes(rt, src, t);
    } finally {
      destroy(src);
    }
    if (p.masterTensor !== null) {
      const srcF32 = createFromArray(rt, data, [...t.shape], 'f32');
      try {
        copyContiguousTensorBytes(rt, srcF32, p.masterTensor);
      } finally {
        destroy(srcF32);
      }
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

export type LoadGptModelFromCheckpointOptions = {
  /** Dtype of parameter buffers; default **`f32`**. */
  readonly computeDtype?: FloatDType;
  /**
   * When the checkpoint stores **`bf16`** / **`fp16`** tensors, set **`true`** to load them into
   * an **`f32`** model (host decode then **`f32`** upload).
   */
  readonly allowDtypeUpcast?: boolean;
};

/**
 * Builds a **`GptModel`** from disk weights (**bin** + **JSON** sidecar).
 *
 * **`f32`** on-disk tensors; **`computeDtype: 'bf16'`** / **`'fp16'`** narrows weights on load.
 * **`allowDtypeUpcast`**: load **16-bit** checkpoints into an **`f32`** model.
 */
export async function loadGptModelFromCheckpoint(
  rt: TensorRuntime,
  binPath: string,
  jsonPath: string,
  options?: LoadGptModelFromCheckpointOptions,
): Promise<{ readonly model: GptModel; readonly config: GptConfig }> {
  const { sidecar, tensors } = await loadGptCheckpointTensors(binPath, jsonPath);
  const config = parseGptConfig(sidecar.config);
  const computeDtype = options?.computeDtype ?? 'f32';
  assertCheckpointCompatibleWithComputeDtype(sidecar, computeDtype, options?.allowDtypeUpcast);

  const model = createGptModel(rt, config, { dtype: computeDtype });
  hydrateGptModelWeights(rt, model, tensors);
  return { model, config };
}
