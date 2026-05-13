import React, { useEffect, useRef } from 'react';
import {
  Box,
  ScrollView,
  Text,
  TextInput,
  render,
  type IScrollViewHandle,
  useStdout,
} from '@wolf-tui/react';

import { createGptKvCache, destroyGptKvCache } from '@llm-transformer-diy/model';
import { generateAutoregressive } from '@llm-transformer-diy/inference';
import { loadGptModelFromCheckpoint } from '@llm-transformer-diy/training';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';

import { resolvePackedCheckpointPaths } from '../paths';
import { createTensorRuntime } from '../runtime';
import { primeTuiTerminalCanvas } from '../tui-prime';

type ChatVm = {
  committed: string;
  pendingUserLine: string | null;
  streamText: string;
  busy: boolean;
  newTokens: number;
  tokensPerSec: number;
  errorLine: string | null;
};

type ChatUiProps = {
  readonly vm: ChatVm;
  readonly onSubmitLine: (line: string) => void;
};

function ChatUi(props: ChatUiProps): React.ReactElement {
  const stdoutCtx = useStdout();
  const termRows = stdoutCtx?.stdout.rows ?? 24;
  const viewportH = Math.max(6, termRows - 8);

  const scrollRef = useRef<IScrollViewHandle | null>(null);

  const transcript = (() => {
    if (!props.vm.busy || props.vm.pendingUserLine === null || props.vm.pendingUserLine === '') {
      return props.vm.committed;
    }
    return `${props.vm.committed}User: ${props.vm.pendingUserLine}\nAssistant: ${props.vm.streamText}`;
  })();

  useEffect(() => {
    scrollRef.current?.scrollToBottom();
  }, [transcript]);

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', padding: 1, flexGrow: 1 }}>
      <Text style={{ fontWeight: 'bold' }}>Chat</Text>
      <Text style={{ color: 'gray' }}>
        {props.vm.busy
          ? `generating · new tokens ${String(props.vm.newTokens)} · tok/s ${props.vm.tokensPerSec.toFixed(1)} · Ctrl+C to quit`
          : `ready · Ctrl+C to quit`}
      </Text>
      {props.vm.errorLine !== null && props.vm.errorLine !== '' ? (
        <Text style={{ color: 'red' }}>{props.vm.errorLine}</Text>
      ) : null}
      <ScrollView ref={scrollRef} height={viewportH} keyBindings={false}>
        <Text>{transcript.length > 0 ? transcript : '[empty — type below]'}</Text>
      </ScrollView>
      <Box style={{ marginTop: 1 }} />
      <TextInput
        isDisabled={props.vm.busy}
        placeholder={props.vm.busy ? '(wait…)' : 'Message — Enter to send'}
        onSubmit={(line) => {
          props.onSubmitLine(line.trim());
        }}
      />
    </Box>
  );
}

export async function runChatCommand(opts: {
  readonly modelArg: string;
  readonly tokenizerPath: string;
  readonly maxNewTokens: number;
  readonly temperature: number;
  readonly topK: number;
  readonly repetitionPenalty?: number;
  readonly repetitionWindow?: number;
}): Promise<void> {
  const rt = await createTensorRuntime();
  if (rt.ctx.isLikelyCpuSoftwareAdapter) {
    console.error(
      'chat: WebGPU is using a software adapter (e.g. LLVMpipe). Use a discrete GPU runtime for speed; see train TUI note on Vulkan/ICD.',
    );
  }
  const { binPath, jsonPath } = resolvePackedCheckpointPaths(opts.modelArg);
  const { model, config } = await loadGptModelFromCheckpoint(rt, binPath, jsonPath);

  const tok = new BPETokenizer();
  await tok.load(opts.tokenizerPath);
  if (tok.vocabSize !== config.vocabSize) {
    throw new Error(
      `chat: tokenizer vocab ${String(tok.vocabSize)} ≠ checkpoint vocab ${String(config.vocabSize)}`,
    );
  }

  const vm: ChatVm = {
    committed: '',
    pendingUserLine: null,
    streamText: '',
    busy: false,
    newTokens: 0,
    tokensPerSec: 0,
    errorLine: null,
  };

  primeTuiTerminalCanvas();

  function buildUi(): React.ReactElement {
    return (
      <ChatUi
        vm={vm}
        onSubmitLine={(line) => {
          void submitLine(line);
        }}
      />
    );
  }

  async function submitLine(trimmedLine: string): Promise<void> {
    if (vm.busy || trimmedLine === '') return;

    const encodePrefix = `${vm.committed}User: ${trimmedLine}\nAssistant: `;
    const promptIdsArr = tok.encode(encodePrefix);
    if (promptIdsArr.length === 0) {
      vm.errorLine = 'chat: encode produced zero tokens — try non-empty text.';
      paint();
      return;
    }

    if (promptIdsArr.length + opts.maxNewTokens > model.config.maxSeqLen) {
      vm.errorLine = `chat: prompt (${String(promptIdsArr.length)} tokens) + --max-new-tokens (${String(opts.maxNewTokens)}) exceeds maxSeqLen ${String(model.config.maxSeqLen)}`;
      vm.pendingUserLine = null;
      vm.streamText = '';
      paint();
      return;
    }

    vm.errorLine = null;
    vm.busy = true;
    vm.pendingUserLine = trimmedLine;
    vm.streamText = '';
    vm.newTokens = 0;
    vm.tokensPerSec = 0;
    paint();

    const promptIds = Uint32Array.from(promptIdsArr);
    const promptRendered = tok.decode([...promptIds], { utf8: 'replace' });
    const generatedIds: number[] = [];
    const tStartGen = performance.now();
    let assistantOut = '';
    const cache = createGptKvCache(rt, model);

    try {
      await generateAutoregressive(rt, model, promptIds, cache, {
        maxNewTokens: opts.maxNewTokens,
        sample: { temperature: opts.temperature, topK: opts.topK },
        ...(opts.repetitionPenalty !== undefined
          ? { repetitionPenalty: opts.repetitionPenalty }
          : {}),
        ...(opts.repetitionWindow !== undefined ? { repetitionWindow: opts.repetitionWindow } : {}),
        onSampledToken: (tid: number) => {
          generatedIds.push(tid);
          const fullDecoded = tok.decode([...promptIds, ...generatedIds], { utf8: 'replace' });
          assistantOut = fullDecoded.slice(promptRendered.length);
          vm.streamText = assistantOut;
          vm.newTokens = generatedIds.length;
          const dt = (performance.now() - tStartGen) / 1000;
          vm.tokensPerSec = vm.newTokens / Math.max(dt, 1e-6);
          paint();
        },
      });
      vm.committed = `${vm.committed}User: ${trimmedLine}\nAssistant: ${assistantOut}\n`;
    } catch (e) {
      vm.errorLine = e instanceof Error ? e.message : String(e);
    } finally {
      destroyGptKvCache(cache);
      vm.busy = false;
      vm.pendingUserLine = null;
      vm.streamText = '';
      vm.newTokens = 0;
      vm.tokensPerSec = 0;
      paint();
    }
  }

  function paint(): void {
    inst.rerender(buildUi());
  }

  const inst = render(buildUi(), { exitOnCtrlC: true, patchConsole: true });
  await inst.waitUntilExit();
}
