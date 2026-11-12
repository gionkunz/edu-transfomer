import React from 'react';
import { render } from '@wolf-tui/react';

import { destroyGptKvCache, type GptKVCache } from '@llm-transformer-diy/model';
import {
  explainSampling,
  mostFocusedHead,
  readLessonHeadKv,
  runLessonForward,
  sampleLogitsWithProbs,
  type LessonHeadKv,
  type LessonTrace,
  type SamplingBreakdown,
} from '@llm-transformer-diy/inference';
import { loadGptModelFromCheckpoint } from '@llm-transformer-diy/training';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';
import type { FloatDType } from '@llm-transformer-diy/tensor';

import { assertRuntimeSupportsComputeDtype } from '../compute-dtype';
import { resolvePackedCheckpointPaths, resolveTokenizerPath } from '../paths';
import { createTensorRuntime } from '../runtime';
import { primeTuiTerminalCanvas } from '../tui-prime';
import { SAMPLING_TOP_N } from './learn-chapter-sampling';
import { LESSON_CHAPTERS } from './learn-chapters';
import { tokenLabel } from './learn-draw';
import { buildLessonMeta, type LessonMeta } from './learn-lesson';
import {
  initialLessonState,
  lessonReducer,
  type LessonAction,
  type LessonState,
} from './learn-state';
import { LearnBusyView, LearnView, type LearnVm } from './learn-view';

/** Prompt used when none is given: short, and common in children's-story corpora. */
export const DEFAULT_LEARN_PROMPT = 'Once upon a time';

/** Milliseconds per animation frame. */
const FRAME_INTERVAL_MS = 140;

/** The lesson is driven with single key presses; refuse to start where that cannot work. */
export function assertInteractiveTerminal(io: {
  readonly stdinIsTTY: boolean;
  readonly stdoutIsTTY: boolean;
}): void {
  if (!io.stdinIsTTY || !io.stdoutIsTTY) {
    throw new Error(
      'learn: needs an interactive terminal (it is driven with the keyboard). ' +
        'Run it directly in a terminal, not through a pipe or redirect.',
    );
  }
}

/**
 * The lesson explains the token after the prompt, so the prompt must leave room
 * for one more position in the model's context.
 */
export function assertLessonPromptFits(promptTokens: number, maxSeqLen: number): void {
  if (promptTokens === 0) {
    throw new Error('learn: the prompt is empty after tokenization');
  }
  const limit = maxSeqLen - 1;
  if (promptTokens > limit) {
    throw new Error(
      `learn: the prompt is ${String(promptTokens)} tokens long; this model can explain prompts of up to ${String(limit)} tokens (context ${String(maxSeqLen)})`,
    );
  }
}

type LessonRun = {
  readonly promptIds: number[];
  readonly trace: LessonTrace;
  readonly cache: GptKVCache;
  readonly meta: LessonMeta;
  readonly tokenLabels: string[];
  /** Keys/values per `layer:head`, filled as heads are visited. */
  readonly kv: Map<string, LessonHeadKv>;
  readonly explained: Map<string, SamplingBreakdown>;
};

export async function runLearnCommand(opts: {
  readonly modelArg: string;
  /** `--tokenizer` as given; when omitted, `tokenizer.json` in the checkpoint dir is used. */
  readonly tokenizerArg?: string;
  readonly prompt: string;
  readonly temperature: number;
  readonly topK: number;
  readonly animation: boolean;
  readonly computeDtype?: FloatDType;
  readonly allowDtypeUpcast?: boolean;
}): Promise<void> {
  assertInteractiveTerminal({
    stdinIsTTY: process.stdin.isTTY === true,
    stdoutIsTTY: process.stdout.isTTY === true,
  });
  // Resolved only now, so a missing terminal is reported before any path problem.
  let tokenizerPath: string;
  try {
    tokenizerPath = resolveTokenizerPath(opts.modelArg, opts.tokenizerArg);
  } catch (err) {
    throw err instanceof Error ? new Error(`learn: ${err.message}`) : err;
  }

  const rt = await createTensorRuntime();
  const dtype = opts.computeDtype ?? 'f32';
  assertRuntimeSupportsComputeDtype(rt.ctx, dtype, 'learn');
  const { binPath, jsonPath } = resolvePackedCheckpointPaths(opts.modelArg);
  const { model, config } = await loadGptModelFromCheckpoint(rt, binPath, jsonPath, {
    computeDtype: dtype,
    ...(opts.allowDtypeUpcast === true ? { allowDtypeUpcast: true } : {}),
  });

  const tok = new BPETokenizer();
  await tok.load(tokenizerPath);
  if (tok.vocabSize !== config.vocabSize) {
    throw new Error(
      `learn: tokenizer vocab ${String(tok.vocabSize)} ≠ checkpoint vocab ${String(config.vocabSize)}`,
    );
  }
  const firstPromptIds = [...tok.encode(opts.prompt)];
  assertLessonPromptFits(firstPromptIds.length, config.maxSeqLen);

  const labelCache = new Map<number, string>();
  const labelForId = (id: number): string => {
    const cached = labelCache.get(id);
    if (cached !== undefined) return cached;
    const piece = tok.decode([id], { utf8: 'replace' });
    // A lone byte of a multi-byte character has no text of its own: show its id.
    const label = /^�*$/.test(piece) ? `‹${String(id)}›` : tokenLabel(piece);
    labelCache.set(id, label);
    return label;
  };

  const rng = Math.random;
  const draw = (trace: LessonTrace, temperature: number, topK: number): number =>
    sampleLogitsWithProbs(rng, trace.rawLogits, { temperature, topK }).tokenId;

  // One GPU job at a time: a key/value readback must not overlap the next forward.
  let gpuQueue: Promise<unknown> = Promise.resolve();
  const onGpu = <T,>(job: () => Promise<T>): Promise<T> => {
    const result = gpuQueue.then(job);
    gpuQueue = result.catch(() => undefined);
    return result;
  };

  const startRun = async (promptIds: number[]): Promise<LessonRun> => {
    const { trace, cache } = await onGpu(() =>
      runLessonForward(rt, model, Uint32Array.from(promptIds)),
    );
    return {
      promptIds,
      trace,
      cache,
      meta: buildLessonMeta(
        LESSON_CHAPTERS,
        { nLayers: config.nLayers, promptLength: promptIds.length },
        { nHeads: config.nHeads, cliTopK: opts.topK, noAnimation: !opts.animation },
      ),
      tokenLabels: promptIds.map(labelForId),
      kv: new Map(),
      explained: new Map(),
    };
  };

  primeTuiTerminalCanvas();

  let run: LessonRun | null = null;
  let state: LessonState | null = null;
  let busy: string | null = 'Running the model on your prompt …';
  let quitting = false;
  let timer: ReturnType<typeof setInterval> | null = null;
  let resolveDone: () => void = () => undefined;
  const done = new Promise<void>((resolve) => {
    resolveDone = resolve;
  });

  const vm = (): LearnVm | null => {
    if (run === null || state === null) return null;
    const current = run;
    const lessonState = state;
    const explain = (temperature: number, topK: number): SamplingBreakdown => {
      const key = `${temperature.toFixed(1)}:${String(topK)}`;
      let breakdown = current.explained.get(key);
      if (breakdown === undefined) {
        breakdown = explainSampling(current.trace.rawLogits, { temperature, topK }, SAMPLING_TOP_N);
        current.explained.set(key, breakdown);
      }
      return breakdown;
    };
    return {
      chapters: LESSON_CHAPTERS,
      meta: current.meta,
      state: lessonState,
      context: {
        trace: current.trace,
        kv:
          current.kv.get(`${String(lessonState.layerIndex)}:${String(lessonState.headIndex)}`) ??
          null,
        tokenLabels: current.tokenLabels,
        labelForId,
        vocabSize: config.vocabSize,
        state: lessonState,
        explain,
      },
      // The extended prompt must itself leave room for a next token.
      canContinue: current.promptIds.length + 1 <= config.maxSeqLen - 1,
      busy,
    };
  };

  const quit = (): void => {
    if (quitting) return;
    quitting = true;
    resolveDone();
  };

  const view = (): React.ReactElement => {
    const current = vm();
    if (current === null) {
      return <LearnBusyView text={busy ?? ''} onQuit={quit} />;
    }
    return (
      <LearnView
        vm={current}
        onAction={dispatch}
        onRedraw={redraw}
        onContinue={() => {
          void continueWithDrawnToken();
        }}
        onQuit={quit}
      />
    );
  };

  const inst = render(view(), { exitOnCtrlC: false, patchConsole: true });
  const paint = (): void => {
    if (!quitting) inst.rerender(view());
  };

  /** Keep the frame timer running exactly while an animation is playing. */
  const syncTimer = (): void => {
    const playing = state?.playing === true && busy === null && !quitting;
    if (playing && timer === null) {
      timer = setInterval(() => {
        dispatch({ type: 'tick' });
      }, FRAME_INTERVAL_MS);
    } else if (!playing && timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  };

  /** Read the selected head's keys and values the first time it is shown. */
  const ensureHeadKv = (): void => {
    if (run === null || state === null) return;
    const current = run;
    const { layerIndex, headIndex } = state;
    const key = `${String(layerIndex)}:${String(headIndex)}`;
    if (current.kv.has(key)) return;
    void onGpu(() => readLessonHeadKv(rt, current.cache, layerIndex, headIndex))
      .then((kv) => {
        current.kv.set(key, kv);
        if (run === current) paint();
      })
      .catch(() => undefined);
  };

  function dispatch(action: LessonAction): void {
    if (run === null || state === null || quitting) return;
    const before = state;
    let next = lessonReducer(before, action, run.meta);
    if (next.temperature !== before.temperature || next.topK !== before.topK) {
      // The old draw may no longer be possible under the new settings.
      next = lessonReducer(
        next,
        { type: 'drawn', tokenId: draw(run.trace, next.temperature, next.topK) },
        run.meta,
      );
    }
    state = next;
    ensureHeadKv();
    syncTimer();
    if (next !== before) paint();
  }

  function redraw(): void {
    if (run === null || state === null) return;
    dispatch({ type: 'drawn', tokenId: draw(run.trace, state.temperature, state.topK) });
  }

  async function continueWithDrawnToken(): Promise<void> {
    if (run === null || state === null || busy !== null) return;
    const previous = run;
    const previousState = state;
    const promptIds = [...previous.promptIds, previousState.drawnTokenId];
    busy = `Running the model with «${labelForId(previousState.drawnTokenId)}» appended …`;
    syncTimer();
    paint();
    try {
      const nextRun = await startRun(promptIds);
      // Queued behind any key/value readback still using the old cache.
      await onGpu(() => {
        destroyGptKvCache(previous.cache);
        return Promise.resolve();
      });
      run = nextRun;
      state = lessonReducer(
        {
          ...previousState,
          drawnTokenId: draw(nextRun.trace, previousState.temperature, previousState.topK),
          drawCount: 0,
        },
        { type: 'restart' },
        nextRun.meta,
      );
    } finally {
      busy = null;
    }
    ensureHeadKv();
    syncTimer();
    paint();
  }

  const releaseCache = (): void => {
    if (run !== null) destroyGptKvCache(run.cache);
  };

  const onResize = (): void => {
    paint();
  };
  process.stdout.on('resize', onResize);

  try {
    const firstRun = await startRun(firstPromptIds);
    run = firstRun;
    const layerIndex = config.nLayers - 1;
    state = initialLessonState(firstRun.meta, {
      layerIndex,
      headIndex: mostFocusedHead(firstRun.trace, layerIndex),
      temperature: opts.temperature,
      topK: opts.topK,
      drawnTokenId: draw(firstRun.trace, opts.temperature, opts.topK),
    });
    busy = null;
    ensureHeadKv();
    syncTimer();
    paint();
    await done;
  } finally {
    process.stdout.off('resize', onResize);
    if (timer !== null) clearInterval(timer);
    quitting = true;
    await gpuQueue.catch(() => undefined);
    releaseCache();
    inst.unmount();
  }
  process.exit(0);
}
