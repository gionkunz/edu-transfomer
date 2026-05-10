import {
  readPackedCheckpointFiles,
  writePackedCheckpointFiles,
  type PackedCheckpointSidecar,
  type TensorPayload,
} from '@llm-transformer-diy/io';
import {
  collectGptParameters,
  gptParameterTensorNames,
  type GptModel,
} from '@llm-transformer-diy/model';
import { toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';

export type SaveGptCheckpointOptions = {
  readonly rt: TensorRuntime;
  readonly model: GptModel;
  readonly binPath: string;
  readonly jsonPath: string;
  /** Stored verbatim as **`sidecar.config`** (e.g. `GptConfig`). */
  readonly config: unknown;
  readonly trainingStep?: number;
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
