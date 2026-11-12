import type { FlowTrace } from '@llm-transformer-diy/inference';

import type { FlowReveal, FlowVariant } from './generate-panels';
import { flowPanelContentRows } from './generate-panels';
import { fmtPercent } from './learn-draw';
import type { Chapter, LessonContext, LessonDims, StepView } from './learn-lesson';

/**
 * Chapter "The stack of blocks": the architecture diagram of `llm generate`,
 * built up along the residual stream — entry, what each block adds, and the
 * model's best guess after each block.
 */

/** Rows the panel adds around its content: border (2) and top margin (1). */
export const FLOW_PANEL_FRAME_ROWS = 3;

type Stage = 'stream' | 'blocks' | 'guess';

/**
 * How to draw the diagram in **`rows`** rows: two rows per block when that
 * fits, else one, else one row per block for a window of blocks around the
 * focused one.
 */
export function stackLayout(
  nLayers: number,
  focusedLayerIndex: number,
  rows: number,
): { readonly variant: FlowVariant; readonly first: number; readonly count: number } {
  const content = rows - FLOW_PANEL_FRAME_ROWS;
  if (flowPanelContentRows(nLayers, 'boxed') <= content) {
    return { variant: 'boxed', first: 0, count: nLayers };
  }
  if (flowPanelContentRows(nLayers, 'compact') <= content) {
    return { variant: 'compact', first: 0, count: nLayers };
  }
  const count = Math.max(1, content - flowPanelContentRows(0, 'compact'));
  const first = Math.max(0, Math.min(nLayers - count, focusedLayerIndex - Math.floor(count / 2)));
  return { variant: 'compact', first, count };
}

/** What is visible at **`frame`** of a stage; the last frame of `guess` is the whole diagram. */
export function stackReveal(stage: Stage, frame: number, nLayers: number): FlowReveal {
  if (stage === 'stream') {
    return { steps: 0, lens: 0, output: false };
  }
  if (stage === 'blocks') {
    return { steps: Math.min(2 * nLayers, frame + 1), lens: 0, output: false };
  }
  const lens = Math.min(nLayers, frame + 1);
  return { steps: 2 * nLayers, lens, output: lens >= nLayers };
}

function windowFlow(flow: FlowTrace, first: number, count: number): FlowTrace {
  return { ...flow, layers: flow.layers.slice(first, first + count) };
}

function explain(ctx: LessonContext, stage: Stage, flow: FlowTrace, windowNote: string): string {
  const { trace, state } = ctx;
  const token = ctx.tokenLabels[trace.promptLength - 1] ?? '';
  switch (stage) {
    case 'stream':
      return (
        `«${token}» enters as a list of ${String(trace.dModel)} numbers: its token vector plus the ` +
        `vector for position ${String(trace.promptLength - 1)}. This running list is the residual ` +
        `stream. No block replaces it; every block only adds to it.${windowNote}`
      );
    case 'blocks':
      return (
        `Each block normalises a copy of the stream, lets attention gather information from ` +
        `earlier tokens and adds the result (⊕). Then a feed-forward layer works on that and adds ` +
        `its result too. +% is the size of each addition relative to the stream.${windowNote}`
      );
    case 'guess': {
      const last = flow.layers[flow.layers.length - 1];
      const top = last !== undefined ? ctx.labelForId(last.topTokenId) : '';
      const p = last !== undefined ? fmtPercent(last.topProbability) : '';
      return (
        `If the model stopped after a block and read the stream out, this is the token it would ` +
        `say. Early blocks guess vaguely, later ones settle: after block ${String(trace.nLayers)} ` +
        `it is «${top}» at ${p}. The bar follows «${ctx.labelForId(state.drawnTokenId)}», the token ` +
        `drawn in the next chapter.${windowNote}`
      );
    }
  }
}

function view(stage: Stage) {
  return (ctx: LessonContext, frame: number): StepView => {
    const { trace, state } = ctx;
    const flow = trace.flowFor(state.drawnTokenId);
    const layout = stackLayout(trace.nLayers, state.layerIndex, ctx.rows);
    const windowed = layout.count < trace.nLayers;
    const windowNote = windowed
      ? ` Blocks ${String(layout.first + 1)}–${String(layout.first + layout.count)} of ${String(trace.nLayers)} shown.`
      : '';
    const final = ctx.explain(state.temperature, state.topK).final;
    return {
      kind: 'flow',
      data: {
        flow: windowFlow(flow, layout.first, layout.count),
        focusedLayerIndex: state.layerIndex,
        queryLabel: ctx.tokenLabels[trace.promptLength - 1] ?? '',
        queryPosition: trace.promptLength - 1,
        sampledLabel: ctx.labelForId(state.drawnTokenId),
        sampledProbability: final[state.drawnTokenId] ?? 0,
        temperature: state.temperature,
        topK: state.topK,
        dModel: trace.dModel,
        vocabSize: ctx.vocabSize,
      },
      variant: layout.variant,
      reveal: stackReveal(stage, frame, trace.nLayers),
      explanation: explain(ctx, stage, flow, windowNote),
    };
  };
}

export const stackChapter: Chapter = {
  id: 'stack',
  title: 'The stack of blocks',
  selectsHead: false,
  samples: false,
  steps: [
    { id: 'stream', title: 'the stream', dial: 'block', frames: () => 1, render: view('stream') },
    {
      id: 'blocks',
      title: 'what a block adds',
      dial: 'block',
      frames: (dims: LessonDims) => 2 * dims.nLayers,
      render: view('blocks'),
    },
    {
      id: 'guess',
      title: 'the guess forms',
      dial: 'block',
      frames: (dims: LessonDims) => dims.nLayers,
      render: view('guess'),
    },
  ],
};
