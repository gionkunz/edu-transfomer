import React from 'react';
import { Box, Text, render, useInput } from '@wolf-tui/react';

import { createGptKvCache, destroyGptKvCache } from '@llm-transformer-diy/model';
import { generateAutoregressive } from '@llm-transformer-diy/inference';
import { loadGptModelFromCheckpoint } from '@llm-transformer-diy/training';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';

import { resolvePackedCheckpointPaths } from '../paths.js';
import { createTensorRuntime } from '../runtime.js';
import { primeTuiTerminalCanvas } from '../tui-prime.js';

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
  readonly repetitionPenalty?: number;
  readonly repetitionWindow?: number;
}): Promise<void> {
  const rt = await createTensorRuntime();
  if (rt.ctx.isLikelyCpuSoftwareAdapter) {
    console.error(
      'generate: WebGPU is using a software adapter (e.g. LLVMpipe). Use a discrete GPU runtime for speed; see train TUI note on Vulkan/ICD.',
    );
  }
  const { binPath, jsonPath } = resolvePackedCheckpointPaths(opts.modelArg);
  const { model, config } = await loadGptModelFromCheckpoint(rt, binPath, jsonPath);

  const tok = new BPETokenizer();
  await tok.load(opts.tokenizerPath);
  if (tok.vocabSize !== config.vocabSize) {
    throw new Error(
      `generate: tokenizer vocab ${String(tok.vocabSize)} ≠ checkpoint vocab ${String(config.vocabSize)}`,
    );
  }

  /** BPE merges are tracked in **bytes**; streaming must not call **`decode`** on single ids (often ill‑formed UTF‑8 fragments). */
  const promptIds = tok.encode(opts.prompt);
  const promptRendered = tok.decode(promptIds, { utf8: 'replace' });
  const cache = createGptKvCache(rt, model);

  const generatedIds: number[] = [];
  let textOut = '';
  let newTok = 0;
  const tStart = performance.now();

  primeTuiTerminalCanvas();
  const inst = render(<GenView text={textOut} newTokens={0} tokensPerSec={0} />, {
    exitOnCtrlC: true,
    patchConsole: true,
  });

  try {
    await generateAutoregressive(rt, model, Uint32Array.from(promptIds), cache, {
      maxNewTokens: opts.maxNewTokens,
      sample: { temperature: opts.temperature, topK: opts.topK },
      ...(opts.repetitionPenalty !== undefined
        ? { repetitionPenalty: opts.repetitionPenalty }
        : {}),
      ...(opts.repetitionWindow !== undefined ? { repetitionWindow: opts.repetitionWindow } : {}),
      onSampledToken: (id: number) => {
        generatedIds.push(id);
        const fullDecoded = tok.decode([...promptIds, ...generatedIds], { utf8: 'replace' });
        textOut = fullDecoded.slice(promptRendered.length);
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
