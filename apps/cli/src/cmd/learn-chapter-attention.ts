import { maxAbsComponent, visibleTokenWindow } from '@llm-transformer-diy/inference';

import {
  fmtPercent,
  fmtShare,
  fmtSigned,
  shareBar,
  signedBar,
  span,
  stripWidth,
  vectorStrip,
} from './learn-draw';
import type { Chapter, LessonContext, LessonDims, Line, Span, StepView } from './learn-lesson';

/**
 * Chapter "One attention head": query → keys → scores → weights → values.
 * One row per prompt token; each step adds a column to the same table.
 */

/** Token rows the table shows at most (fits the 80×24 layout with all extra lines). */
export const ATTENTION_MAX_TOKEN_ROWS = 9;
const STRIP_MAX = 24;
const LABEL_WIDTH = 10;
const SCORE_BAR_WIDTH = 11;
const WEIGHT_BAR_WIDTH = 8;
/** Frames of the "grow in" animations. */
const GROW_FRAMES = 6;

type Stage = 'query' | 'keys' | 'scores' | 'weights' | 'mix';
const STAGE_ORDER: readonly Stage[] = ['query', 'keys', 'scores', 'weights', 'mix'];

function visibleRows(dims: LessonDims): number {
  return Math.min(dims.promptLength, ATTENTION_MAX_TOKEN_ROWS);
}

/** Token rows that fit: the table needs 6 further rows (query, gap, header, hidden, gap, output). */
function tokenRowBudget(ctx: LessonContext): number {
  return Math.max(1, Math.min(ATTENTION_MAX_TOKEN_ROWS, ctx.rows - 6));
}

function reached(stage: Stage, current: Stage): boolean {
  return STAGE_ORDER.indexOf(current) >= STAGE_ORDER.indexOf(stage);
}

function renderTable(ctx: LessonContext, stage: Stage, frame: number, frames: number): Line[] {
  const { trace, state, kv } = ctx;
  const head = trace.head(state.layerIndex, state.headIndex);
  const window = visibleTokenWindow(head.weights, tokenRowBudget(ctx));
  const shown = window.indices;
  const lastPos = trace.promptLength - 1;
  const done = frame >= frames - 1;
  const grow = frames <= 1 ? 1 : Math.min(1, frame / (frames - 1));

  const strip = stripWidth(trace.dK, STRIP_MAX);
  const keysShown = shown.map((i) => kv?.keys[i]).filter((v) => v !== undefined);
  const valuesShown = shown.map((i) => kv?.values[i]).filter((v) => v !== undefined);
  // One brightness scale for query and keys so matching cells can be compared by eye.
  const qkScale = maxAbsComponent([head.query, ...keysShown]);
  const valueScale = maxAbsComponent([head.mixed, ...valuesShown]);
  let scoreScale = 0;
  for (const i of shown) scoreScale = Math.max(scoreScale, Math.abs(head.scores[i] ?? 0));

  // Order in which the mix step picks rows up: heaviest first.
  const byWeight = [...shown].sort((a, b) => (head.weights[b] ?? 0) - (head.weights[a] ?? 0));

  const lines: Line[] = [];
  // The query sits in the same column as the keys below it, so cells line up.
  const queryLabel = ctx.tokenLabels[lastPos] ?? '';
  lines.push([
    span('  query ▸'.padEnd(LABEL_WIDTH + 9, ' '), 'yellow'),
    ...vectorStrip(head.query, strip, qkScale, stage === 'query' ? grow : 1),
    span('  ← query of ', 'gray'),
    span(`«${queryLabel}»`, undefined, true),
    span(` · ${String(trace.dK)} numbers`, 'gray'),
  ]);
  lines.push([]);

  const stripTitle = stage === 'mix' ? 'value' : 'key';
  const header: Span[] = [
    span(`  ${'token'.padEnd(LABEL_WIDTH + 2, ' ')} pos `, 'gray'),
    span((reached('keys', stage) ? stripTitle : '').padEnd(strip + 2, ' '), 'gray'),
  ];
  if (reached('keys', stage)) {
    header.push(span('score'.padEnd(SCORE_BAR_WIDTH + 8, ' '), 'gray'));
  }
  if (reached('weights', stage)) {
    header.push(span('weight', 'gray'));
  }
  lines.push(header);

  shown.forEach((pos, rowIndex) => {
    const label = `«${ctx.tokenLabels[pos] ?? ''}»`.padEnd(LABEL_WIDTH + 2, ' ');
    const isQueryToken = pos === lastPos;
    // keys step: rows are compared one after another; `frame` is the row being compared.
    const comparing = stage === 'keys' && !done && rowIndex === frame;
    const compared = stage !== 'keys' || done || rowIndex < frame;
    // mix step: rows join the mix heaviest first.
    const mixRank = byWeight.indexOf(pos);
    const mixing = stage === 'mix' && !done && mixRank === frame;
    const mixed = stage !== 'mix' || done || mixRank <= frame;
    const active = comparing || mixing;

    const row: Span[] = [
      span(active ? '▶ ' : '  ', 'yellow'),
      span(label, active ? 'yellow' : isQueryToken ? undefined : 'gray', isQueryToken || active),
      span(` ${String(pos).padStart(3, ' ')} `, 'gray'),
    ];

    if (!reached('keys', stage)) {
      lines.push(row);
      return;
    }

    const vec = stage === 'mix' ? kv?.values[pos] : kv?.keys[pos];
    if (vec === undefined) {
      row.push(span('…'.padEnd(strip, ' '), 'gray'));
    } else if (stage === 'mix') {
      row.push(
        ...(mixed
          ? vectorStrip(vec, strip, valueScale)
          : [span('·'.repeat(Math.min(strip, vec.length)).padEnd(strip, ' '), 'gray')]),
      );
    } else if (compared || comparing) {
      row.push(...vectorStrip(vec, strip, qkScale));
    } else {
      row.push(span(' '.repeat(strip)));
    }
    row.push(span('  '));

    const score = head.scores[pos] ?? 0;
    if (stage === 'keys') {
      // The number appears as the row is compared; the bars come in the next step.
      row.push(
        span(' '.repeat(SCORE_BAR_WIDTH + 1)),
        span(
          compared || comparing ? fmtSigned(score) : ' '.repeat(6),
          comparing ? 'yellow' : undefined,
        ),
      );
    } else {
      const drawn = stage === 'scores' ? score * grow : score;
      row.push(...signedBar(drawn, scoreScale, SCORE_BAR_WIDTH), span(' '), span(fmtSigned(score)));
    }

    if (reached('weights', stage)) {
      const w = head.weights[pos] ?? 0;
      const drawn = stage === 'weights' ? w * grow : w;
      row.push(
        span('  '),
        span(shareBar(drawn, WEIGHT_BAR_WIDTH), pos === head.peakIndex ? 'yellow' : 'cyan'),
        span(` ${fmtShare(w)}`, pos === head.peakIndex ? 'yellow' : undefined),
      );
    }
    lines.push(row);
  });

  if (window.hiddenCount > 0) {
    const share = reached('weights', stage)
      ? `; together they hold ${fmtShare(window.hiddenWeight)} of the weight`
      : '';
    lines.push([
      span(
        `  … ${String(window.hiddenCount)} earlier token${window.hiddenCount === 1 ? '' : 's'} not shown${share}`,
        'gray',
      ),
    ]);
  }

  if (stage === 'mix') {
    lines.push([]);
    lines.push([
      span('  output ▸'.padEnd(LABEL_WIDTH + 9, ' '), 'yellow'),
      ...(done
        ? vectorStrip(head.mixed, strip, valueScale)
        : [span('…'.padEnd(strip, ' '), 'gray')]),
      span(done ? '  = Σ weight × value → stream' : '', 'gray'),
    ]);
  }
  return lines;
}

function peakInfo(ctx: LessonContext): {
  label: string;
  pos: number;
  score: number;
  weight: number;
} {
  const head = ctx.trace.head(ctx.state.layerIndex, ctx.state.headIndex);
  const pos = head.peakIndex;
  return {
    label: ctx.tokenLabels[pos] ?? '',
    pos,
    score: head.scores[pos] ?? 0,
    weight: head.weights[pos] ?? 0,
  };
}

function explain(ctx: LessonContext, stage: Stage): string {
  const { trace, state } = ctx;
  const block = String(state.layerIndex + 1);
  const token = ctx.tokenLabels[trace.promptLength - 1] ?? '';
  const peak = peakInfo(ctx);
  switch (stage) {
    case 'query':
      return (
        `Block ${block} reads the stream for «${token}», and each of its ${String(trace.nHeads)} heads ` +
        `turns it into a short "query" of ${String(trace.dK)} numbers. The query is this head's ` +
        `question: which earlier token would help me here? Cyan cells are positive, magenta ` +
        `negative, brighter is larger.`
      );
    case 'keys':
      return (
        `Every token left a "key" behind when it was processed: its answer to "what do I ` +
        `offer?". The head multiplies query and key cell by cell and adds it all up. Cells with ` +
        `the same colour push the score up, opposite colours pull it down.`
      );
    case 'scores':
      return (
        `One score per token: that sum, divided by √${String(trace.dK)} to keep the numbers in a ` +
        `useful range. The best match is «${peak.label}» at position ${String(peak.pos)} with ` +
        `${peak.score.toFixed(2)}. Scores can be negative; only their differences matter.`
      );
    case 'weights':
      return (
        `Softmax turns the scores into shares that add up to 1: each score is exponentiated and ` +
        `divided by the total, so a small lead becomes a large share. «${peak.label}» gets ` +
        `${fmtPercent(peak.weight)} of this head's attention.`
      );
    case 'mix':
      return (
        `Each token also left a "value": what it hands over when it is attended to. The head's ` +
        `output is the values averaged with the weights, so it is mostly «${peak.label}»'s value. ` +
        `The outputs of all ${String(trace.nHeads)} heads are combined and added to the stream.`
      );
  }
}

function view(stage: Stage, frames: (dims: LessonDims) => number) {
  return (ctx: LessonContext, frame: number): StepView => ({
    kind: 'lines',
    lines: renderTable(
      ctx,
      stage,
      frame,
      frames({ nLayers: ctx.trace.nLayers, promptLength: ctx.trace.promptLength }),
    ),
    explanation: explain(ctx, stage),
  });
}

const growFrames = (): number => GROW_FRAMES;
const rowFrames = (dims: LessonDims): number => visibleRows(dims) + 1;

export const attentionChapter: Chapter = {
  id: 'attention',
  title: 'One attention head',
  selectsHead: true,
  samples: false,
  steps: [
    {
      id: 'query',
      title: 'the query',
      dial: 'block',
      frames: growFrames,
      render: view('query', growFrames),
    },
    {
      id: 'keys',
      title: 'compare with keys',
      dial: 'block',
      frames: rowFrames,
      render: view('keys', rowFrames),
    },
    {
      id: 'scores',
      title: 'scores',
      dial: 'block',
      frames: growFrames,
      render: view('scores', growFrames),
    },
    {
      id: 'weights',
      title: 'weights',
      dial: 'block',
      frames: growFrames,
      render: view('weights', growFrames),
    },
    {
      id: 'mix',
      title: 'mix the values',
      dial: 'block',
      frames: rowFrames,
      render: view('mix', rowFrames),
    },
  ],
};
