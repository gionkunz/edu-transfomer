import type { SamplingBreakdown, SamplingBreakdownRow } from '@llm-transformer-diy/inference';

import { fmtPercent, fmtShare, shareBar, span } from './learn-draw';
import type { Chapter, LessonContext, Line, Span, StepView } from './learn-lesson';

/**
 * Chapter "Picking the next token": the same candidate list followed through
 * the sampler — raw probabilities, temperature, top-k, the draw.
 */

/** Candidates listed (most likely first). */
export const SAMPLING_TOP_N = 10;
const LABEL_WIDTH = 12;
const BAR_WIDTH = 36;
const MORPH_FRAMES = 6;
const DRAW_FRAMES = 9;

type Stage = 'raw' | 'temperature' | 'topk' | 'draw';
const STAGES: readonly Stage[] = ['raw', 'temperature', 'topk', 'draw'];

function stageValue(row: SamplingBreakdownRow, stage: Stage | 'zero'): number {
  switch (stage) {
    case 'zero':
      return 0;
    case 'raw':
      return row.rawProbability;
    case 'temperature':
      return row.temperedProbability;
    case 'topk':
    case 'draw':
      return row.finalProbability;
  }
}

function othersValue(b: SamplingBreakdown, stage: Stage | 'zero'): number {
  switch (stage) {
    case 'zero':
      return 0;
    case 'raw':
      return b.others.rawProbability;
    case 'temperature':
      return b.others.temperedProbability;
    case 'topk':
    case 'draw':
      return b.others.finalProbability;
  }
}

/** The stage a step's bars start from when it is entered: the picture of the step before. */
function previousStage(stage: Stage): Stage | 'zero' {
  const at = STAGES.indexOf(stage);
  return at <= 0 ? 'zero' : (STAGES[at - 1] ?? 'zero');
}

function samplerLabel(temperature: number, topK: number): string {
  return `temperature ${temperature.toFixed(1)} · top-k ${topK > 0 ? String(topK) : 'off'}`;
}

function pipelineLine(stage: Stage): Line {
  const names: [Stage, string][] = [
    ['raw', 'scores → probabilities'],
    ['temperature', 'temperature'],
    ['topk', 'top-k'],
    ['draw', 'draw'],
  ];
  const out: Span[] = [span('  ')];
  names.forEach(([s, name], i) => {
    if (i > 0) out.push(span('  →  ', 'gray'));
    out.push(s === stage ? span(name, 'yellow', true) : span(name, 'gray'));
  });
  return out;
}

function renderList(ctx: LessonContext, stage: Stage, frame: number, frames: number): Line[] {
  const { state } = ctx;
  const now = ctx.explain(state.temperature, state.topK);
  const done = frame >= frames - 1;
  const t = frames <= 1 ? 1 : Math.min(1, frame / (frames - 1));

  // Bars move from an earlier picture to this one: after a temperature / top-k
  // change from the same stage under the old settings, otherwise from the
  // previous stage. The numbers always show the current values.
  const changed = state.samplerFrom;
  const before = changed !== null ? ctx.explain(changed.temperature, changed.topK) : now;
  const fromStage: Stage | 'zero' = changed !== null ? stage : previousStage(stage);
  const beforeById = new Map(before.rows.map((r) => [r.tokenId, r]));
  // The draw step keeps its bars still and animates the marker instead.
  const morph = stage === 'draw' && changed === null ? 1 : t;

  const keptRows = now.rows.filter((r) => r.finalProbability > 0);
  const drawnRow = now.rows.findIndex((r) => r.tokenId === state.drawnTokenId);
  // While the draw runs the marker hops over the candidates, then lands.
  const marker =
    stage !== 'draw'
      ? -1
      : done || keptRows.length <= 1
        ? drawnRow
        : now.rows.findIndex((r) => r === keptRows[(frame + state.drawCount) % keptRows.length]);

  const lines: Line[] = [pipelineLine(stage), []];
  now.rows.forEach((row, i) => {
    const to = stageValue(row, stage);
    const fromRow = beforeById.get(row.tokenId);
    const from = fromRow !== undefined ? stageValue(fromRow, fromStage) : 0;
    const shownBar = from + (to - from) * morph;
    const removed = (stage === 'topk' || stage === 'draw') && !row.kept;
    const isMarked = i === marker;
    const landed = isMarked && done;
    const color = removed ? 'gray' : isMarked ? 'yellow' : 'cyan';
    const label = `«${ctx.labelForId(row.tokenId)}»`.padEnd(LABEL_WIDTH + 2, ' ');
    const line: Span[] = [
      span(isMarked ? '▶ ' : '  ', 'yellow'),
      span(label, removed ? 'gray' : landed ? 'yellow' : undefined, landed),
      span(' '),
      span(shareBar(shownBar, BAR_WIDTH), color),
      span(` ${fmtShare(to)}`, removed ? 'gray' : undefined),
    ];
    if (removed) {
      line.push(span('  removed', 'gray'));
    } else if (landed) {
      line.push(span('  ◀ drawn', 'yellow', true));
    }
    lines.push(line);
  });

  const others = othersValue(now, stage);
  const othersFrom = othersValue(before, fromStage);
  const othersCount = Math.max(0, ctx.vocabSize - now.rows.length);
  lines.push([
    span('  '),
    span(`${String(othersCount)} others`.padEnd(LABEL_WIDTH + 2, ' '), 'gray'),
    span(' '),
    span(shareBar(othersFrom + (others - othersFrom) * morph, BAR_WIDTH), 'gray'),
    span(` ${fmtShare(others)}`, 'gray'),
  ]);

  const status: Span[] = [span(`  ${samplerLabel(state.temperature, state.topK)}`, 'cyan')];
  if ((stage === 'topk' || stage === 'draw') && state.topK > 0) {
    status.push(
      span(`  ·  ${String(now.keptCount)} of ${String(ctx.vocabSize)} tokens kept`, 'gray'),
    );
  }
  if (stage === 'draw' && done && drawnRow < 0) {
    status.push(
      span('  ·  drawn: ', 'gray'),
      span(`«${ctx.labelForId(state.drawnTokenId)}»`, 'yellow', true),
      span(
        ` ${fmtShare(now.final[state.drawnTokenId] ?? 0)} (below the top ${String(now.rows.length)})`,
        'gray',
      ),
    );
  }
  lines.push([], status);
  return lines;
}

function explain(ctx: LessonContext, stage: Stage): string {
  const { state } = ctx;
  const now = ctx.explain(state.temperature, state.topK);
  const V = String(ctx.vocabSize);
  switch (stage) {
    case 'raw':
      return (
        `The final stream is compared with all ${V} tokens of the vocabulary, giving one score ` +
        `each; softmax turns the scores into probabilities. These are the model's own beliefs. ` +
        `The top ${String(now.rows.length)} are shown; all other tokens share ` +
        `${fmtPercent(now.others.rawProbability)}.`
      );
    case 'temperature':
      if (state.temperature <= 1e-8) {
        return (
          `Temperature 0 switches chance off: the most likely token is always taken, so the ` +
          `same prompt always gives the same text. Press ↑ to raise it and let other candidates ` +
          `back in.`
        );
      }
      return (
        `Temperature divides every score before softmax. Below 1 the leader gains and the tail ` +
        `shrinks: safer, more repetitive text. Above 1 everything flattens: more surprising, ` +
        `more mistakes. It is ${state.temperature.toFixed(1)} now; ↑ and ↓ change it.`
      );
    case 'topk':
      if (state.topK <= 0) {
        return (
          `Top-k is off, so all ${V} tokens stay in the draw, including very unlikely ones. ` +
          `Press k to keep only the 5, 10 or 40 best: the rest is removed and the survivors are ` +
          `rescaled to add up to 1.`
        );
      }
      return (
        `Top-k keeps only the ${String(state.topK)} highest-scoring tokens and removes all others, ` +
        `so one unlucky draw cannot derail the text. The survivors are rescaled to add up to 1. ` +
        `Press k to try other settings.`
      );
    case 'draw': {
      const p = now.final[state.drawnTokenId] ?? 0;
      return (
        `Finally one token is drawn at random, each with the chance shown: this time ` +
        `«${ctx.labelForId(state.drawnTokenId)}» (${fmtPercent(p)}). Press r to draw again; that ` +
        `randomness is why one prompt gives many texts. The token is appended and everything ` +
        `starts over.`
      );
    }
  }
}

function view(stage: Stage, frames: number) {
  return (ctx: LessonContext, frame: number): StepView => ({
    kind: 'lines',
    lines: renderList(ctx, stage, frame, frames),
    explanation: explain(ctx, stage),
  });
}

function step(stage: Stage, title: string, frames: number, restartOnDraw = false) {
  return {
    id: stage,
    title,
    dial: 'temperature' as const,
    frames: () => frames,
    restartOnDraw,
    render: view(stage, frames),
  };
}

export const samplingChapter: Chapter = {
  id: 'sampling',
  title: 'Picking the next token',
  selectsHead: false,
  samples: true,
  steps: [
    step('raw', 'what the model believes', MORPH_FRAMES),
    step('temperature', 'temperature', MORPH_FRAMES),
    step('topk', 'top-k', MORPH_FRAMES),
    step('draw', 'the draw', DRAW_FRAMES, true),
  ],
};
