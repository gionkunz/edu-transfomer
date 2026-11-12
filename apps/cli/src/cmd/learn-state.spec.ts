import { describe, expect, it } from 'vitest';

import { fixtureMeta, FIXTURE } from './learn-fixture';
import { isLastStep, overallStepIndex, topKCycleFor, totalSteps } from './learn-lesson';
import {
  initialLessonState,
  lastFrame,
  lessonReducer,
  type LessonAction,
  type LessonState,
} from './learn-state';

const INIT = { layerIndex: 1, headIndex: 0, temperature: 0.8, topK: 40, drawnTokenId: 3 };

function run(
  state: LessonState,
  meta: ReturnType<typeof fixtureMeta>,
  ...actions: LessonAction['type'][]
) {
  return actions.reduce((s, type) => lessonReducer(s, { type } as LessonAction, meta), state);
}

/** Play the current step's animation to its end. */
function finish(state: LessonState, meta: ReturnType<typeof fixtureMeta>): LessonState {
  let s = state;
  for (let i = 0; i < 100 && s.playing; i++) s = lessonReducer(s, { type: 'tick' }, meta);
  return s;
}

describe('lesson structure', () => {
  const meta = fixtureMeta(4);

  it('has three chapters with 5, 3 and 4 steps', () => {
    expect(meta.chapters.map((c) => c.steps.length)).toEqual([5, 3, 4]);
    expect(totalSteps(meta)).toBe(12);
    expect(overallStepIndex(meta, 1, 2)).toBe(7);
    expect(isLastStep(meta, 2, 3)).toBe(true);
    expect(isLastStep(meta, 2, 2)).toBe(false);
  });

  it('top-k cycle contains off and the command-line value once', () => {
    expect(topKCycleFor(40)).toEqual([0, 5, 10, 40]);
    expect(topKCycleFor(25)).toEqual([0, 5, 10, 25, 40]);
    expect(topKCycleFor(0)).toEqual([0, 5, 10, 40]);
  });
});

describe('lessonReducer: navigation', () => {
  const meta = fixtureMeta(4);
  const start = initialLessonState(meta, INIT);

  it('starts on the first step with its animation running', () => {
    expect(start.chapterIndex).toBe(0);
    expect(start.stepIndex).toBe(0);
    expect(start.frame).toBe(0);
    expect(start.playing).toBe(true);
  });

  it('next crosses a chapter boundary, prev goes back to the last step of the chapter before', () => {
    const atEnd = run(start, meta, 'next', 'next', 'next', 'next');
    expect([atEnd.chapterIndex, atEnd.stepIndex]).toEqual([0, 4]);
    const crossed = run(atEnd, meta, 'next');
    expect([crossed.chapterIndex, crossed.stepIndex]).toEqual([1, 0]);
    const back = run(crossed, meta, 'prev');
    expect([back.chapterIndex, back.stepIndex]).toEqual([0, 4]);
  });

  it('stays put at both ends of the lesson', () => {
    expect(run(start, meta, 'prev')).toBe(start);
    let s = start;
    for (let i = 0; i < 30; i++) s = run(s, meta, 'next');
    expect([s.chapterIndex, s.stepIndex]).toEqual([2, 3]);
    expect(run(s, meta, 'next')).toBe(s);
  });

  it('a digit jumps to the first step of that chapter and ignores unknown chapters', () => {
    const jumped = lessonReducer(start, { type: 'jumpChapter', chapterIndex: 2 }, meta);
    expect([jumped.chapterIndex, jumped.stepIndex]).toEqual([2, 0]);
    expect(lessonReducer(start, { type: 'jumpChapter', chapterIndex: 3 }, meta)).toBe(start);
  });

  it('navigating away and back keeps every selection', () => {
    const selected = run(start, meta, 'dialUp', 'nextHead');
    const there = run(selected, meta, 'next', 'next', 'prev', 'prev');
    expect(there.layerIndex).toBe(selected.layerIndex);
    expect(there.headIndex).toBe(selected.headIndex);
    expect(there.temperature).toBe(selected.temperature);
    expect(there.topK).toBe(selected.topK);
    expect(there.drawnTokenId).toBe(selected.drawnTokenId);
  });
});

describe('lessonReducer: animation', () => {
  const meta = fixtureMeta(4);
  const start = initialLessonState(meta, INIT);

  it('ticks to the last frame and stops there', () => {
    const end = finish(start, meta);
    expect(end.frame).toBe(lastFrame(end, meta));
    expect(end.playing).toBe(false);
    expect(lessonReducer(end, { type: 'tick' }, meta)).toBe(end);
  });

  it('space pauses, resumes, and replays a finished animation', () => {
    const one = run(start, meta, 'tick');
    const paused = run(one, meta, 'togglePlay');
    expect(paused.playing).toBe(false);
    expect(run(paused, meta, 'tick').frame).toBe(one.frame);
    expect(run(paused, meta, 'togglePlay').playing).toBe(true);

    const replay = run(finish(start, meta), meta, 'togglePlay');
    expect(replay.frame).toBe(0);
    expect(replay.playing).toBe(true);
  });

  it('entering a step restarts from frame 0, so the old animation cannot continue', () => {
    const mid = run(start, meta, 'tick', 'tick');
    const next = run(mid, meta, 'next');
    expect(next.frame).toBe(0);
    expect(next.stepIndex).toBe(1);
  });

  it('--no-animation shows final frames and ignores space and ticks', () => {
    const still = fixtureMeta(4, true);
    const s = initialLessonState(still, INIT);
    expect(s.frame).toBe(lastFrame(s, still));
    expect(s.playing).toBe(false);
    expect(run(s, still, 'togglePlay')).toBe(s);
    expect(run(s, still, 'tick')).toBe(s);
    const n = run(s, still, 'next');
    expect(n.frame).toBe(lastFrame(n, still));
    expect(n.playing).toBe(false);
  });

  it('the finished frame is the same with and without animation', () => {
    const still = fixtureMeta(4, true);
    const a = finish(start, meta);
    const b = initialLessonState(still, INIT);
    expect(a.frame).toBe(b.frame);
  });
});

describe('lessonReducer: block and head', () => {
  const meta = fixtureMeta(4);
  const start = finish(initialLessonState(meta, INIT), meta);

  it('h/H cycle heads with wrap-around and replay the step', () => {
    const next = run(start, meta, 'nextHead');
    expect(next.headIndex).toBe(1);
    expect(next.frame).toBe(0);
    expect(next.playing).toBe(true);
    expect(run(next, meta, 'nextHead').headIndex).toBe(0);
    expect(run(start, meta, 'prevHead').headIndex).toBe(FIXTURE.nHeads - 1);
  });

  it('up/down move the block with wrap-around; down is the deeper block', () => {
    expect(start.layerIndex).toBe(1);
    expect(run(start, meta, 'dialUp').layerIndex).toBe(0);
    expect(run(start, meta, 'dialDown').layerIndex).toBe(0);
    expect(run(start, meta, 'dialUp', 'dialDown').layerIndex).toBe(1);
  });

  it('the selected block carries into the stack chapter, where h does nothing', () => {
    const moved = run(start, meta, 'dialUp');
    const stack = lessonReducer(moved, { type: 'jumpChapter', chapterIndex: 1 }, meta);
    expect(stack.layerIndex).toBe(0);
    expect(run(stack, meta, 'nextHead')).toBe(stack);
  });
});

describe('lessonReducer: sampler', () => {
  const meta = fixtureMeta(4);
  const sampling = finish(
    lessonReducer(initialLessonState(meta, INIT), { type: 'jumpChapter', chapterIndex: 2 }, meta),
    meta,
  );

  it('up/down change the temperature in steps of 0.1 within [0, 2]', () => {
    const up = run(sampling, meta, 'dialUp');
    expect(up.temperature).toBe(0.9);
    expect(up.samplerFrom).toEqual({ temperature: 0.8, topK: 40 });
    expect(up.frame).toBe(0);
    let low = sampling;
    for (let i = 0; i < 30; i++) low = run(low, meta, 'dialDown');
    expect(low.temperature).toBe(0);
    let high = sampling;
    for (let i = 0; i < 30; i++) high = run(high, meta, 'dialUp');
    expect(high.temperature).toBe(2);
    expect(run(high, meta, 'dialUp')).toBe(high);
  });

  it('k cycles top-k through the list and only in the sampling chapter', () => {
    const seen = [sampling.topK];
    let s = sampling;
    for (let i = 0; i < 4; i++) {
      s = run(s, meta, 'cycleTopK');
      seen.push(s.topK);
    }
    expect(seen).toEqual([40, 0, 5, 10, 40]);
    const attention = initialLessonState(meta, INIT);
    expect(run(attention, meta, 'cycleTopK')).toBe(attention);
  });

  it('leaving a step forgets the sampler transition', () => {
    const changed = run(sampling, meta, 'dialUp');
    expect(run(changed, meta, 'next').samplerFrom).toBeNull();
  });

  it('a draw replays only the draw step', () => {
    const drawn = lessonReducer(sampling, { type: 'drawn', tokenId: 9 }, meta);
    expect(drawn.drawnTokenId).toBe(9);
    expect(drawn.drawCount).toBe(1);
    expect(drawn.frame).toBe(sampling.frame);

    const drawStep = finish(run(sampling, meta, 'next', 'next', 'next'), meta);
    const redrawn = lessonReducer(drawStep, { type: 'drawn', tokenId: 9 }, meta);
    expect(redrawn.frame).toBe(0);
    expect(redrawn.playing).toBe(true);
  });

  it('restart returns to the first step and keeps selections', () => {
    const changed = run(sampling, meta, 'dialUp', 'cycleTopK');
    const again = run(changed, meta, 'restart');
    expect([again.chapterIndex, again.stepIndex]).toEqual([0, 0]);
    expect(again.temperature).toBe(changed.temperature);
    expect(again.topK).toBe(changed.topK);
    expect(again.layerIndex).toBe(changed.layerIndex);
  });
});
