import { readFileSync } from 'node:fs';

import { assertValidGptConfig, type GptConfig } from '@llm-transformer-diy/model';

function tiny(vocabSize: number): GptConfig {
  return {
    vocabSize,
    maxSeqLen: 64,
    dModel: 32,
    nHeads: 4,
    dFf: 64,
    nLayers: 1,
  };
}

/** Small GPT-shaped config (vocab comes from tokenizer). */
function small(vocabSize: number): GptConfig {
  return {
    vocabSize,
    maxSeqLen: 256,
    dModel: 128,
    nHeads: 4,
    dFf: 512,
    nLayers: 2,
  };
}

const PRESETS: Record<string, (v: number) => GptConfig> = {
  tiny,
  small,
};

export function resolveTrainConfig(configArg: string, vocabSize: number): GptConfig {
  const preset = PRESETS[configArg];
  if (preset !== undefined) {
    return preset(vocabSize);
  }
  // JSON file
  return readGptConfigFile(configArg, vocabSize);
}

function readGptConfigFile(filePath: string, vocabSize: number): GptConfig {
  const rawText = readFileSync(filePath, 'utf8');
  const raw: unknown = JSON.parse(rawText) as unknown;
  if (raw === null || typeof raw !== 'object') {
    throw new Error(`--config file must contain a JSON object (${filePath})`);
  }
  const o = raw as Record<string, unknown>;
  const fromFile =
    Number.isFinite(Number(o.vocabSize)) && Number(o.vocabSize) > 0
      ? Number(o.vocabSize)
      : vocabSize;
  if (fromFile !== vocabSize) {
    throw new Error(
      `--config vocabSize ${String(fromFile)} disagrees with tokenizer vocab ${String(vocabSize)}`,
    );
  }
  const base: GptConfig = {
    vocabSize: fromFile,
    maxSeqLen: Number(o.maxSeqLen),
    dModel: Number(o.dModel),
    nHeads: Number(o.nHeads),
    dFf: Number(o.dFf),
    nLayers: Number(o.nLayers),
  };
  assertValidGptConfig(base);
  return base;
}
