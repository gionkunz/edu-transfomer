import { describe, expect, it } from 'vitest';

import { revealedBlockSteps } from './generate-panels';
import { stackLayout, stackReveal } from './learn-chapter-stack';
import { LESSON_CHAPTERS } from './learn-chapters';
import { EXPLANATION_ROWS, LESSON_TEXT_WIDTH, lineText, lineWidth, wrapText } from './learn-draw';
import { FIXTURE, fixtureContext } from './learn-fixture';
import type { LessonContext, StepView } from './learn-lesson';

const ROWS = 15;

function chapter(id: string) {
  const found = LESSON_CHAPTERS.find((c) => c.id === id);
  if (found === undefined) throw new Error(`no chapter ${id}`);
  return found;
}

function step(chapterId: string, stepId: string) {
  const found = chapter(chapterId).steps.find((s) => s.id === stepId);
  if (found === undefined) throw new Error(`no step ${stepId}`);
  return found;
}

function finalView(chapterId: string, stepId: string, ctx: LessonContext): StepView {
  const s = step(chapterId, stepId);
  const frames = s.frames({ nLayers: ctx.trace.nLayers, promptLength: ctx.trace.promptLength });
  return s.render(ctx, frames - 1);
}

function text(view: StepView): string[] {
  if (view.kind !== 'lines') throw new Error('expected a lines view');
  return view.lines.map(lineText);
}

describe('every step at 80×24', () => {
  for (const promptLength of [1, 4, 30]) {
    it(`fits rows and columns with a ${String(promptLength)}-token prompt, on every frame`, () => {
      const { ctx, meta } = fixtureContext(promptLength);
      LESSON_CHAPTERS.forEach((c, ci) => {
        c.steps.forEach((s, si) => {
          const frames = meta.chapters[ci]?.steps[si]?.frames ?? 1;
          for (let f = 0; f < frames; f++) {
            const view = s.render(ctx, f);
            if (view.kind === 'lines') {
              expect(view.lines.length, `${c.id}/${s.id} rows`).toBeLessThanOrEqual(ROWS);
              for (const line of view.lines) {
                expect(lineWidth(line), `${c.id}/${s.id}: ${lineText(line)}`).toBeLessThanOrEqual(
                  LESSON_TEXT_WIDTH,
                );
              }
            }
            const explanation = wrapText(view.explanation, LESSON_TEXT_WIDTH);
            expect(explanation.length, `${c.id}/${s.id} explanation`).toBeGreaterThanOrEqual(1);
            expect(explanation.length, `${c.id}/${s.id}: ${view.explanation}`).toBeLessThanOrEqual(
              EXPLANATION_ROWS,
            );
          }
        });
      });
    });
  }
});

describe('chapter: one attention head', () => {
  it('names the last prompt token as the one being processed', () => {
    const { ctx } = fixtureContext(4);
    const lines = text(finalView('attention', 'query', ctx));
    expect(lines[0]).toContain('query of «tok3»');
    expect(lines[0]).toContain('32 numbers');
    expect(lines.filter((l) => l.includes('«tok')).length).toBe(5);
  });

  it('weights sum to 1 and the highest score has the largest weight', () => {
    const { ctx } = fixtureContext(4);
    const head = ctx.trace.head(ctx.state.layerIndex, ctx.state.headIndex);
    const lines = text(finalView('attention', 'weights', ctx)).filter((l) =>
      /«tok\d»\s+\d/.test(l),
    );
    expect(lines.length).toBe(4);
    const weights = lines.map((l) => Number(l.trim().split(/\s+/).pop()));
    expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 1);
    let best = 0;
    head.scores.forEach((s, i) => {
      if (s > (head.scores[best] ?? -Infinity)) best = i;
    });
    expect(weights.indexOf(Math.max(...weights))).toBe(best);
  });

  it('compares tokens one after another, each getting its score as it is compared', () => {
    const { ctx } = fixtureContext(4);
    const keys = step('attention', 'keys');
    const scoreCount = (frame: number): number =>
      text(keys.render(ctx, frame)).filter((l) => /[+-]\d+\.\d\d\s*$/.test(l)).length;
    expect(scoreCount(0)).toBe(1);
    expect(scoreCount(1)).toBe(2);
    expect(scoreCount(4)).toBe(4);
    expect(text(keys.render(ctx, 1)).filter((l) => l.startsWith('▶')).length).toBe(1);
    expect(text(keys.render(ctx, 4)).filter((l) => l.startsWith('▶')).length).toBe(0);
  });

  it('shows the most recent tokens of a long prompt and totals the hidden weight', () => {
    const { ctx } = fixtureContext(30);
    const lines = text(finalView('attention', 'weights', ctx));
    expect(lines.some((l) => l.includes('«tok29»'))).toBe(true);
    expect(lines.some((l) => l.includes('«tok0»'))).toBe(false);
    const hidden = lines.find((l) => l.includes('earlier tokens not shown'));
    expect(hidden).toMatch(/… 21 earlier tokens not shown; together they hold \S+ of the weight/);
  });

  it('shows a placeholder until the keys of the head have been read', () => {
    const { ctx } = fixtureContext(4);
    const lines = text(finalView('attention', 'keys', { ...ctx, kv: null }));
    expect(lines.filter((l) => l.includes('…')).length).toBe(4);
  });

  it('another head gives another picture', () => {
    const a = fixtureContext(4, { headIndex: 0 }).ctx;
    const b = fixtureContext(4, { headIndex: 1 }).ctx;
    expect(text(finalView('attention', 'weights', a))).not.toEqual(
      text(finalView('attention', 'weights', b)),
    );
  });

  it('the mix step ends with the head output', () => {
    const { ctx } = fixtureContext(4);
    const mix = step('attention', 'mix');
    expect(text(mix.render(ctx, 0)).some((l) => l.includes('Σ weight × value'))).toBe(false);
    expect(
      text(finalView('attention', 'mix', ctx)).some((l) => l.includes('Σ weight × value')),
    ).toBe(true);
  });
});

describe('chapter: the stack of blocks', () => {
  it('reveals the stream, then each block step, then the guesses block by block', () => {
    const L = 3;
    expect(stackReveal('stream', 0, L)).toEqual({ steps: 0, lens: 0, output: false });
    expect(stackReveal('blocks', 0, L)).toEqual({ steps: 1, lens: 0, output: false });
    expect(stackReveal('blocks', 2 * L - 1, L)).toEqual({ steps: 2 * L, lens: 0, output: false });
    expect(stackReveal('guess', 0, L)).toEqual({ steps: 2 * L, lens: 1, output: false });
    expect(stackReveal('guess', L - 1, L)).toEqual({ steps: 2 * L, lens: L, output: true });
  });

  it('a reveal of everything draws the same blocks as no reveal', () => {
    const full = stackReveal('guess', 2, 3);
    for (let l = 0; l < 3; l++) {
      expect(revealedBlockSteps(full, l)).toBe(revealedBlockSteps(undefined, l));
    }
    expect(revealedBlockSteps({ steps: 3, lens: 0, output: false }, 0)).toBe(2);
    expect(revealedBlockSteps({ steps: 3, lens: 0, output: false }, 1)).toBe(1);
    expect(revealedBlockSteps({ steps: 3, lens: 0, output: false }, 2)).toBe(0);
  });

  it('uses two rows per block when they fit, one otherwise, and a window for deep models', () => {
    expect(stackLayout(2, 0, ROWS)).toEqual({ variant: 'boxed', first: 0, count: 2 });
    expect(stackLayout(8, 0, ROWS)).toEqual({ variant: 'compact', first: 0, count: 8 });
    const deep = stackLayout(24, 12, ROWS);
    expect(deep.variant).toBe('compact');
    expect(deep.count).toBe(9);
    expect(deep.first).toBeLessThanOrEqual(12);
    expect(deep.first + deep.count).toBeGreaterThan(12);
    expect(stackLayout(24, 23, ROWS).first).toBe(15);
  });

  it('marks the selected block and ends on the guess that the sampling chapter starts from', () => {
    const { ctx } = fixtureContext(4, { layerIndex: 0 });
    const view = finalView('stack', 'guess', ctx);
    if (view.kind !== 'flow') throw new Error('expected the flow view');
    expect(view.data.focusedLayerIndex).toBe(0);
    expect(view.reveal).toEqual({
      steps: 2 * FIXTURE.nLayers,
      lens: FIXTURE.nLayers,
      output: true,
    });
    const lastBlock = view.data.flow.layers[FIXTURE.nLayers - 1];
    const rawTop = ctx.explain(1, 0).rows[0];
    expect(lastBlock?.topTokenId).toBe(rawTop?.tokenId);
    expect(lastBlock?.topProbability).toBeCloseTo(rawTop?.rawProbability ?? -1, 5);
  });
});

describe('chapter: picking the next token', () => {
  const probabilityOfFirstRow = (lines: string[]): number =>
    Number(/(\d\.\d\d|<\.01)(\s|$)/.exec(lines[2] ?? '')?.[1]?.replace('<', '0') ?? NaN);

  it('lowering the temperature does not lower the top token', () => {
    let previous = 0;
    for (const temperature of [1.6, 1.2, 0.8, 0.4, 0]) {
      const { ctx } = fixtureContext(4, { temperature });
      const p = probabilityOfFirstRow(text(finalView('sampling', 'temperature', ctx)));
      expect(p).toBeGreaterThanOrEqual(previous);
      previous = p;
    }
    expect(previous).toBe(1);
  });

  it('top-k marks everything outside the kept set as removed', () => {
    const { ctx } = fixtureContext(4, { topK: 5 });
    const lines = text(finalView('sampling', 'topk', ctx));
    expect(lines.filter((l) => l.includes('removed')).length).toBe(5);
    expect(lines.some((l) => l.includes('5 of 40 tokens kept'))).toBe(true);
    const off = text(finalView('sampling', 'topk', fixtureContext(4, { topK: 0 }).ctx));
    expect(off.some((l) => l.includes('removed'))).toBe(false);
  });

  it('marks the drawn token, which is one of the kept candidates', () => {
    const { ctx } = fixtureContext(4, { topK: 5 });
    const kept = ctx.explain(0.8, 5).rows.filter((r) => r.kept);
    const drawn = kept[2];
    if (drawn === undefined) throw new Error('fixture has fewer than 3 kept rows');
    const lines = text(
      finalView('sampling', 'draw', {
        ...ctx,
        state: { ...ctx.state, drawnTokenId: drawn.tokenId },
      }),
    );
    const marked = lines.filter((l) => l.includes('◀ drawn'));
    expect(marked.length).toBe(1);
    expect(marked[0]).toContain(`«v${String(drawn.tokenId)}»`);
    expect(marked[0]).not.toContain('removed');
  });

  it('names a drawn token that is not among the listed candidates', () => {
    const { ctx } = fixtureContext(4, { topK: 0, temperature: 2 });
    const listed = new Set(ctx.explain(2, 0).rows.map((r) => r.tokenId));
    let outsider = 0;
    while (listed.has(outsider)) outsider += 1;
    const lines = text(
      finalView('sampling', 'draw', { ...ctx, state: { ...ctx.state, drawnTokenId: outsider } }),
    );
    expect(lines.some((l) => l.includes(`drawn: «v${String(outsider)}»`))).toBe(true);
  });

  it('bars move from the old to the new distribution while numbers show the new one', () => {
    const { ctx } = fixtureContext(4, {
      temperature: 0.2,
      samplerFrom: { temperature: 2, topK: 40 },
    });
    const s = step('sampling', 'temperature');
    const bar = (frame: number): number =>
      (text(s.render(ctx, frame))[2] ?? '').replace(/[^█▏▎▍▌▋▊▉]/g, '').length;
    expect(bar(0)).toBeLessThan(bar(5));
    expect(probabilityOfFirstRow(text(s.render(ctx, 0)))).toBe(
      probabilityOfFirstRow(text(s.render(ctx, 5))),
    );
  });
});
