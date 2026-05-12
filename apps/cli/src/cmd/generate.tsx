import React from 'react';
import { Box, Text, render, useInput } from '@wolf-tui/react';

import { createGptKvCache, destroyGptKvCache } from '@llm-transformer-diy/model';
import { generateAutoregressive } from '@llm-transformer-diy/inference';
import { loadGptModelFromCheckpoint } from '@llm-transformer-diy/training';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';

import { resolvePackedCheckpointPaths } from '../paths.js';
import { createTensorRuntime } from '../runtime.js';

type GenViewProps = {
  readonly text: string;
  readonly newTokens: number;
  readonly tokensPerSec: number;
};

function GenView(props: GenViewProps): React.ReactElement {
  useInput((input, key) => {
    if (input === 'q') {
      process.exit(0);
    }
    if (key.ctrl && input === 'c') {
      process.exit(0);
    }
  });

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', padding: 1 }}>
      <Text style={{ fontWeight: 'bold' }}>Generate</Text>
      <Text
        style={{ color: 'gray' }}
      >{`new tokens ${String(props.newTokens)} · tok/s ${props.tokensPerSec.toFixed(1)} — q to quit`}</Text>
      <Text>{props.text}</Text>
    </Box>
  );
}

export async function runGenerateCommand(opts: {
  readonly modelArg: string;
  readonly tokenizerPath: string;
  readonly prompt: string;
  readonly maxNewTokens: number;
  readonly temperature: number;
  readonly topK: number;
}): Promise<void> {
  const rt = await createTensorRuntime();
  const { binPath, jsonPath } = resolvePackedCheckpointPaths(opts.modelArg);
  const { model, config } = await loadGptModelFromCheckpoint(rt, binPath, jsonPath);

  const tok = new BPETokenizer();
  await tok.load(opts.tokenizerPath);
  if (tok.vocabSize !== config.vocabSize) {
    throw new Error(
      `generate: tokenizer vocab ${String(tok.vocabSize)} ≠ checkpoint vocab ${String(config.vocabSize)}`,
    );
  }

  const promptIds = Uint32Array.from(tok.encode(opts.prompt));
  const cache = createGptKvCache(rt, model);

  let textOut = '';
  let newTok = 0;
  const tStart = performance.now();

  const inst = render(<GenView text={textOut} newTokens={0} tokensPerSec={0} />, {
    exitOnCtrlC: true,
    patchConsole: true,
  });

  try {
    await generateAutoregressive(rt, model, promptIds, cache, {
      maxNewTokens: opts.maxNewTokens,
      sample: { temperature: opts.temperature, topK: opts.topK },
      onSampledToken: (id: number) => {
        textOut += tok.decode([id]);
        newTok += 1;
        const dt = (performance.now() - tStart) / 1000;
        inst.rerender(
          <GenView text={textOut} newTokens={newTok} tokensPerSec={newTok / Math.max(dt, 1e-6)} />,
        );
      },
    });
  } finally {
    destroyGptKvCache(cache);
    inst.unmount();
  }
}
