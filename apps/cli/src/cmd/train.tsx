import { copyFile, mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import React, { useMemo } from 'react';
import { Box, Text, render, useInput } from '@wolf-tui/react';

import { createParameterRegistry } from '@llm-transformer-diy/autograd';
import { discoverTrainingFiles } from '@llm-transformer-diy/io';
import { collectGptParameters, createGptModel } from '@llm-transformer-diy/model';
import {
  AdamW,
  Trainer,
  createLmBatch,
  getCosineLr,
  registerModelParameters,
  saveGptCheckpoint,
} from '@llm-transformer-diy/training';
import { destroy } from '@llm-transformer-diy/tensor';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';

import { resolveTrainConfig } from '../config-presets.js';
import { createTensorRuntime } from '../runtime.js';

type TrainViewProps = {
  readonly label: string;
  readonly step: number;
  readonly totalSteps: number;
  readonly loss: number;
  readonly lr: number;
  readonly tokensPerSec: number;
  readonly elapsedSec: number;
};

function TrainView(props: TrainViewProps): React.ReactElement {
  useInput((input, key) => {
    if (input === 'q') {
      process.exit(0);
    }
    if (key.ctrl && input === 'c') {
      process.exit(0);
    }
  });

  const lossStr = useMemo(() => props.loss.toFixed(4), [props.loss]);
  const lrStr = useMemo(() => props.lr.toExponential(3), [props.lr]);

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', padding: 1 }}>
      <Text style={{ fontWeight: 'bold' }}>{`Training — ${props.label}`}</Text>
      <Text>{`Step ${String(props.step)} / ${String(props.totalSteps)}`}</Text>
      <Text>{`Loss ${lossStr}`}</Text>
      <Text>{`LR ${lrStr}`}</Text>
      <Text>{`Tokens/s (approx) ${props.tokensPerSec.toFixed(0)}`}</Text>
      <Text
        style={{ color: 'gray' }}
      >{`Elapsed ${props.elapsedSec.toFixed(0)}s — press q to quit`}</Text>
    </Box>
  );
}

export async function runTrainCommand(opts: {
  readonly dataGlobs: string[];
  readonly outDir: string;
  readonly tokenizerPath: string;
  readonly configArg: string;
  readonly steps: number;
  readonly seqLen: number;
}): Promise<void> {
  const rt = await createTensorRuntime();
  const tokenizer = new BPETokenizer();
  await tokenizer.load(opts.tokenizerPath);
  const vocabSize = tokenizer.vocabSize;
  const gptCfg = resolveTrainConfig(opts.configArg, vocabSize);

  const files = await discoverTrainingFiles(opts.dataGlobs);
  if (files.length === 0) {
    throw new Error('train: --data matched no files');
  }
  const textParts: string[] = [];
  for (const rel of files) {
    textParts.push(await readFile(path.resolve(rel), 'utf8'));
  }
  const corpus = textParts.join('\n');
  const tokenIds = Uint32Array.from(tokenizer.encode(corpus));
  if (tokenIds.length <= opts.seqLen) {
    throw new Error(
      `train: need more tokens than seqLen (${String(tokenIds.length)} <= ${String(opts.seqLen)})`,
    );
  }

  const model = createGptModel(rt, gptCfg);
  const reg = createParameterRegistry();
  registerModelParameters(reg, collectGptParameters(model));
  const opt = new AdamW(rt, reg, { lr: 3e-4, weightDecay: 0.1 });

  const totalSteps = opts.steps;
  const schedule = {
    lrMax: 3e-4,
    lrMin: 1e-5,
    warmupSteps: Math.min(10, Math.max(1, Math.floor(totalSteps / 10))),
    totalSteps,
  };

  const trainer = new Trainer(rt, model, reg, opt, {
    maxGradNorm: 1.0,
    schedule,
  });

  const label = `${opts.configArg} · ${String(gptCfg.nLayers)}L · ${String(model.config.dModel)}d`;

  let stepUi = 0;
  let lossUi = 0;
  let lrUi = schedule.lrMax;
  let tpsUi = 0;
  const tStart = Date.now();

  const inst = render(
    <TrainView
      label={label}
      step={stepUi}
      totalSteps={totalSteps}
      loss={lossUi}
      lr={lrUi}
      tokensPerSec={tpsUi}
      elapsedSec={0}
    />,
    { exitOnCtrlC: true, patchConsole: true },
  );

  try {
    for (let s = 0; s < totalSteps; s++) {
      const maxStart = tokenIds.length - opts.seqLen - 1;
      const start = Math.floor((s * 9973) % (maxStart + 1));
      const { input, targets } = createLmBatch(rt, tokenIds, start, opts.seqLen);
      const t0 = performance.now();
      const loss = await trainer.trainStep(input, targets, s);
      const dt = Math.max(1e-6, (performance.now() - t0) / 1000);
      destroy(input);
      destroy(targets);

      stepUi = s + 1;
      lossUi = loss;
      lrUi = getCosineLr(s, schedule);
      tpsUi = opts.seqLen / dt;

      inst.rerender(
        <TrainView
          label={label}
          step={stepUi}
          totalSteps={totalSteps}
          loss={lossUi}
          lr={lrUi}
          tokensPerSec={tpsUi}
          elapsedSec={(Date.now() - tStart) / 1000}
        />,
      );
    }

    const outAbs = path.resolve(opts.outDir);
    await mkdir(outAbs, { recursive: true });
    const binPath = path.join(outAbs, 'model.bin');
    const jsonPath = path.join(outAbs, 'model.json');
    await saveGptCheckpoint({
      rt,
      model,
      binPath,
      jsonPath,
      config: gptCfg,
      trainingStep: totalSteps,
    });
    await copyFile(opts.tokenizerPath, path.join(outAbs, 'tokenizer.json'));
  } finally {
    inst.unmount();
    opt.dispose();
  }
}
