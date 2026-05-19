import { writePackedCheckpointFiles, readPackedCheckpointFiles } from '@llm-transformer-diy/io';
import { gptParameterTensorNames, type GptConfig, type GptModel } from '@llm-transformer-diy/model';
import { type FloatDType, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import type { AdamW } from './adamw';
import { saveGptCheckpoint, type SaveGptCheckpointOptions } from './checkpoint';
import type { CosineLrSchedule } from './scheduler';

export const TRAIN_STATE_KIND = 'llm-transformer-diy/train-state';
export const TRAIN_STATE_VERSION = 1;

/** Default periodic checkpoint interval for long runs (`0` = final save only). */
export const TRAIN_DEFAULT_CHECKPOINT_EVERY = 2000;

export type TrainingCheckpointPaths = {
  readonly modelBin: string;
  readonly modelJson: string;
  readonly optimizerBin: string;
  readonly optimizerJson: string;
  readonly trainStateJson: string;
  readonly tokenizerJson: string;
};

export type TrainStateSidecar = {
  readonly kind: typeof TRAIN_STATE_KIND;
  readonly version: typeof TRAIN_STATE_VERSION;
  /** Completed optimizer steps (resume loop starts at this index). */
  readonly step: number;
  readonly totalSteps: number;
  readonly seqLen: number;
  readonly computeDtype: FloatDType;
  readonly schedule: CosineLrSchedule;
  readonly weightDecay: number;
  readonly maxGradNorm: number;
  readonly trainLossLast?: number;
};

export type SaveTrainingCheckpointOptions = {
  readonly rt: TensorRuntime;
  readonly model: GptModel;
  readonly optimizer: AdamW;
  readonly outDir: string;
  readonly config: GptConfig;
  readonly completedStep: number;
  readonly totalSteps: number;
  readonly seqLen: number;
  readonly computeDtype: FloatDType;
  readonly schedule: CosineLrSchedule;
  readonly weightDecay: number;
  readonly maxGradNorm: number;
  readonly trainLossLast?: number;
  readonly tokenizerSourcePath?: string;
  readonly onCheckpointTensor?: SaveGptCheckpointOptions['onCheckpointTensor'];
};

export type ResumeTrainingCheckpoint = {
  readonly completedStep: number;
  readonly trainState: TrainStateSidecar;
};

/** Resolve canonical artifact paths under **`outDir`**. */
export function trainingCheckpointPaths(outDir: string): TrainingCheckpointPaths {
  return {
    modelBin: path.join(outDir, 'model.bin'),
    modelJson: path.join(outDir, 'model.json'),
    optimizerBin: path.join(outDir, 'optimizer.bin'),
    optimizerJson: path.join(outDir, 'optimizer.json'),
    trainStateJson: path.join(outDir, 'train-state.json'),
    tokenizerJson: path.join(outDir, 'tokenizer.json'),
  };
}

/**
 * Validates **`checkpointEvery`** for **`llm train --checkpoint-every`**.
 * **`0`** disables mid-run saves (final checkpoint still written).
 */
export function resolveCheckpointEvery(totalSteps: number, checkpointEvery: number): number {
  if (!Number.isInteger(totalSteps) || totalSteps <= 0) {
    throw new Error(
      `resolveCheckpointEvery: totalSteps must be a positive integer (got ${String(totalSteps)})`,
    );
  }
  if (!Number.isInteger(checkpointEvery) || checkpointEvery < 0) {
    throw new Error(
      `train: --checkpoint-every must be a non-negative integer (got ${String(checkpointEvery)})`,
    );
  }
  return checkpointEvery;
}

function configsEqual(a: GptConfig, b: GptConfig): boolean {
  return (
    a.vocabSize === b.vocabSize &&
    a.maxSeqLen === b.maxSeqLen &&
    a.dModel === b.dModel &&
    a.nHeads === b.nHeads &&
    a.dFf === b.dFf &&
    a.nLayers === b.nLayers
  );
}

function parseTrainStateSidecar(raw: unknown): TrainStateSidecar {
  if (raw === null || typeof raw !== 'object') {
    throw new Error('train checkpoint: train-state.json root must be an object');
  }
  const o = raw as Record<string, unknown>;
  if (o.kind !== TRAIN_STATE_KIND) {
    throw new Error(
      `train checkpoint: train-state.json kind must be "${TRAIN_STATE_KIND}" (got ${JSON.stringify(o.kind)})`,
    );
  }
  if (o.version !== TRAIN_STATE_VERSION) {
    throw new Error(
      `train checkpoint: unsupported train-state version ${String(o.version)} (expected ${String(TRAIN_STATE_VERSION)})`,
    );
  }
  const scheduleRaw = o.schedule;
  if (scheduleRaw === null || typeof scheduleRaw !== 'object') {
    throw new Error('train checkpoint: train-state.json missing schedule object');
  }
  const s = scheduleRaw as Record<string, unknown>;
  const schedule: CosineLrSchedule = {
    lrMax: Number(s.lrMax),
    lrMin: Number(s.lrMin),
    warmupSteps: Number(s.warmupSteps),
    totalSteps: Number(s.totalSteps),
  };
  const state: TrainStateSidecar = {
    kind: TRAIN_STATE_KIND,
    version: TRAIN_STATE_VERSION,
    step: Number(o.step),
    totalSteps: Number(o.totalSteps),
    seqLen: Number(o.seqLen),
    computeDtype: o.computeDtype === 'bf16' ? 'bf16' : 'f32',
    schedule,
    weightDecay: Number(o.weightDecay),
    maxGradNorm: Number(o.maxGradNorm),
    ...(o.trainLossLast !== undefined ? { trainLossLast: Number(o.trainLossLast) } : {}),
  };
  if (
    !Number.isInteger(state.step) ||
    state.step < 0 ||
    !Number.isInteger(state.totalSteps) ||
    state.totalSteps <= 0 ||
    !Number.isInteger(state.seqLen) ||
    state.seqLen <= 0
  ) {
    throw new Error('train checkpoint: train-state.json has invalid step/totalSteps/seqLen');
  }
  return state;
}

async function writeTrainStateJson(
  trainStatePath: string,
  state: TrainStateSidecar,
): Promise<void> {
  await writeFile(trainStatePath, `${JSON.stringify(state, null, 2)}\n`, 'utf8');
}

async function saveOptimizerCheckpoint(
  rt: TensorRuntime,
  optimizer: AdamW,
  model: GptModel,
  binPath: string,
  jsonPath: string,
): Promise<void> {
  const names = gptParameterTensorNames(model);
  const moments = await optimizer.exportNamedMoments(rt, names);
  const tensors: Record<string, { shape: readonly number[]; data: Float32Array; dtype?: 'f32' }> =
    {};
  for (const name of names) {
    const entry = moments[name];
    if (!entry) {
      throw new Error(`saveOptimizerCheckpoint: missing moments for "${name}"`);
    }
    const n = entry.m.length;
    tensors[`${name}.m`] = { shape: [n], data: entry.m };
    tensors[`${name}.v`] = { shape: [n], data: entry.v };
  }
  await writePackedCheckpointFiles({
    binPath,
    jsonPath,
    config: {
      kind: 'llm-transformer-diy/adamw-moments',
      version: 1,
      timestep: optimizer.getTimestep(),
    },
    tensors,
  });
}

/**
 * Writes model weights, AdamW moments, train-state sidecar, and optionally tokenizer copy.
 */
export async function saveTrainingCheckpoint(opts: SaveTrainingCheckpointOptions): Promise<void> {
  const paths = trainingCheckpointPaths(opts.outDir);
  const trainState: TrainStateSidecar = {
    kind: TRAIN_STATE_KIND,
    version: TRAIN_STATE_VERSION,
    step: opts.completedStep,
    totalSteps: opts.totalSteps,
    seqLen: opts.seqLen,
    computeDtype: opts.computeDtype,
    schedule: opts.schedule,
    weightDecay: opts.weightDecay,
    maxGradNorm: opts.maxGradNorm,
    ...(opts.trainLossLast !== undefined ? { trainLossLast: opts.trainLossLast } : {}),
  };

  await saveGptCheckpoint({
    rt: opts.rt,
    model: opts.model,
    binPath: paths.modelBin,
    jsonPath: paths.modelJson,
    config: opts.config,
    trainingStep: opts.completedStep,
    ...(opts.onCheckpointTensor !== undefined
      ? { onCheckpointTensor: opts.onCheckpointTensor }
      : {}),
  });
  await saveOptimizerCheckpoint(
    opts.rt,
    opts.optimizer,
    opts.model,
    paths.optimizerBin,
    paths.optimizerJson,
  );
  await writeTrainStateJson(paths.trainStateJson, trainState);

  if (opts.tokenizerSourcePath !== undefined) {
    const { copyFile } = await import('node:fs/promises');
    await copyFile(opts.tokenizerSourcePath, paths.tokenizerJson);
  }
}

/** Returns resume metadata when **`--resume`** artifacts exist and match the current run config. */
export async function loadTrainingCheckpointForResume(opts: {
  readonly outDir: string;
  readonly config: GptConfig;
  readonly totalSteps: number;
  readonly seqLen: number;
  readonly computeDtype: FloatDType;
}): Promise<ResumeTrainingCheckpoint> {
  const paths = trainingCheckpointPaths(opts.outDir);
  let trainStateRaw: unknown;
  try {
    trainStateRaw = JSON.parse(await readFile(paths.trainStateJson, 'utf8')) as unknown;
  } catch (e) {
    throw new Error(
      `train --resume: missing or unreadable ${paths.trainStateJson} — run without --resume or ensure a periodic checkpoint exists (${e instanceof Error ? e.message : String(e)})`,
    );
  }
  const trainState = parseTrainStateSidecar(trainStateRaw);

  const { sidecar: modelSidecar } = await readPackedCheckpointFiles(
    paths.modelBin,
    paths.modelJson,
  );
  const modelCfg = modelSidecar.config;
  if (modelCfg === null || typeof modelCfg !== 'object') {
    throw new Error('train --resume: model.json config must be an object');
  }
  const ckptConfig = modelCfg as GptConfig;
  if (!configsEqual(opts.config, ckptConfig)) {
    throw new Error(
      'train --resume: --config / preset does not match checkpoint model.json config (architecture mismatch)',
    );
  }

  if (trainState.totalSteps !== opts.totalSteps) {
    throw new Error(
      `train --resume: --steps ${String(opts.totalSteps)} ≠ checkpoint totalSteps ${String(trainState.totalSteps)}`,
    );
  }
  if (trainState.seqLen !== opts.seqLen) {
    throw new Error(
      `train --resume: --seq-len ${String(opts.seqLen)} ≠ checkpoint seqLen ${String(trainState.seqLen)}`,
    );
  }
  if (trainState.computeDtype !== opts.computeDtype) {
    throw new Error(
      `train --resume: --dtype ${opts.computeDtype} ≠ checkpoint computeDtype ${trainState.computeDtype}`,
    );
  }
  if (trainState.step <= 0) {
    throw new Error('train --resume: checkpoint step is 0 — nothing to resume');
  }
  if (trainState.step >= opts.totalSteps) {
    throw new Error(
      `train --resume: checkpoint already at step ${String(trainState.step)} ≥ --steps ${String(opts.totalSteps)}`,
    );
  }

  return { completedStep: trainState.step, trainState };
}

/** Restores Adam **`t`** and moment buffers from **`optimizer.bin`**. */
export async function hydrateAdamWFromCheckpoint(
  rt: TensorRuntime,
  optimizer: AdamW,
  model: GptModel,
  optimizerBinPath: string,
  optimizerJsonPath: string,
): Promise<void> {
  const { sidecar, tensors } = await readPackedCheckpointFiles(optimizerBinPath, optimizerJsonPath);
  const cfg = sidecar.config;
  if (cfg === null || typeof cfg !== 'object') {
    throw new Error('train --resume: optimizer.json config must be an object');
  }
  const tRaw = (cfg as Record<string, unknown>).timestep;
  const t = Number(tRaw);
  if (!Number.isInteger(t) || t < 0) {
    throw new Error(`train --resume: invalid optimizer timestep (got ${String(tRaw)})`);
  }
  optimizer.setTimestep(t);

  const names = gptParameterTensorNames(model);
  const moments: Record<string, { m: Float32Array; v: Float32Array }> = {};
  for (const name of names) {
    const m = tensors[`${name}.m`];
    const v = tensors[`${name}.v`];
    if (!m || !v) {
      throw new Error(`train --resume: optimizer checkpoint missing moments for "${name}"`);
    }
    moments[name] = { m, v };
  }
  optimizer.hydrateNamedMoments(rt, names, moments);
}

/** Whether a mid-run checkpoint should fire after completing step **`completedStep`**. */
export function shouldSavePeriodicCheckpoint(
  completedStep: number,
  totalSteps: number,
  checkpointEvery: number,
): boolean {
  if (checkpointEvery <= 0) {
    return false;
  }
  if (completedStep >= totalSteps) {
    return false;
  }
  return completedStep % checkpointEvery === 0;
}
