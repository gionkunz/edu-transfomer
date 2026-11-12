import type { LessonMeta } from './learn-lesson';

/** Where the learner is and what they have selected. Plain data; see **`lessonReducer`**. */
export type LessonState = {
  readonly chapterIndex: number;
  readonly stepIndex: number;
  /** Current animation frame of the step, `0 … frames - 1`. */
  readonly frame: number;
  readonly playing: boolean;
  readonly layerIndex: number;
  readonly headIndex: number;
  readonly temperature: number;
  readonly topK: number;
  /** The token the sampler drew; changes with **`drawn`**. */
  readonly drawnTokenId: number;
  /** Counts draws, so two draws of the same token still differ. */
  readonly drawCount: number;
  /**
   * Sampler settings before the last temperature / top-k change, while the
   * step animates from the old distribution to the new one. `null` otherwise.
   */
  readonly samplerFrom: { readonly temperature: number; readonly topK: number } | null;
};

export type LessonAction =
  | { readonly type: 'next' }
  | { readonly type: 'prev' }
  | { readonly type: 'jumpChapter'; readonly chapterIndex: number }
  | { readonly type: 'tick' }
  | { readonly type: 'togglePlay' }
  | { readonly type: 'dialUp' }
  | { readonly type: 'dialDown' }
  | { readonly type: 'nextHead' }
  | { readonly type: 'prevHead' }
  | { readonly type: 'cycleTopK' }
  | { readonly type: 'drawn'; readonly tokenId: number }
  | { readonly type: 'restart' };

export const TEMPERATURE_MIN = 0;
export const TEMPERATURE_MAX = 2;
export const TEMPERATURE_STEP = 0.1;

export function initialLessonState(
  meta: LessonMeta,
  init: {
    readonly layerIndex: number;
    readonly headIndex: number;
    readonly temperature: number;
    readonly topK: number;
    readonly drawnTokenId: number;
  },
): LessonState {
  return restartAnimation(
    {
      chapterIndex: 0,
      stepIndex: 0,
      frame: 0,
      playing: false,
      layerIndex: init.layerIndex,
      headIndex: init.headIndex,
      temperature: init.temperature,
      topK: init.topK,
      drawnTokenId: init.drawnTokenId,
      drawCount: 0,
      samplerFrom: null,
    },
    meta,
  );
}

function stepMeta(
  state: LessonState,
  meta: LessonMeta,
): LessonMeta['chapters'][number]['steps'][number] {
  const step = meta.chapters[state.chapterIndex]?.steps[state.stepIndex];
  if (step === undefined) {
    throw new Error(
      `lesson: no step ${String(state.stepIndex)} in chapter ${String(state.chapterIndex)}`,
    );
  }
  return step;
}

/** Last frame index of the current step. */
export function lastFrame(state: LessonState, meta: LessonMeta): number {
  return stepMeta(state, meta).frames - 1;
}

/** Start the current step's animation over (or jump to its end with `--no-animation`). */
function restartAnimation(state: LessonState, meta: LessonMeta): LessonState {
  const last = lastFrame(state, meta);
  if (meta.noAnimation || last === 0) {
    return { ...state, frame: last, playing: false };
  }
  return { ...state, frame: 0, playing: true };
}

function enterStep(
  state: LessonState,
  meta: LessonMeta,
  chapterIndex: number,
  stepIndex: number,
): LessonState {
  return restartAnimation({ ...state, chapterIndex, stepIndex, samplerFrom: null }, meta);
}

function wrap(value: number, size: number): number {
  return ((value % size) + size) % size;
}

function changeSampler(
  state: LessonState,
  meta: LessonMeta,
  next: { readonly temperature: number; readonly topK: number },
): LessonState {
  if (next.temperature === state.temperature && next.topK === state.topK) {
    return state;
  }
  return restartAnimation(
    {
      ...state,
      ...next,
      samplerFrom: { temperature: state.temperature, topK: state.topK },
    },
    meta,
  );
}

function turnDial(state: LessonState, meta: LessonMeta, direction: 1 | -1): LessonState {
  const { dial } = stepMeta(state, meta);
  if (dial === 'block') {
    // Blocks are drawn top to bottom, so "down" (+1) is the next, deeper block.
    const layerIndex = wrap(state.layerIndex + direction, meta.nLayers);
    if (layerIndex === state.layerIndex) return state;
    const moved = { ...state, layerIndex };
    // The attention chapter shows one head of that block: replay it for the new one.
    return meta.chapters[state.chapterIndex]?.selectsHead === true
      ? restartAnimation(moved, meta)
      : moved;
  }
  if (dial === 'temperature') {
    // "Up" raises the temperature, i.e. direction -1 in block terms.
    const raw = state.temperature - direction * TEMPERATURE_STEP;
    const temperature = Math.min(
      TEMPERATURE_MAX,
      Math.max(TEMPERATURE_MIN, Math.round(raw * 10) / 10),
    );
    return changeSampler(state, meta, { temperature, topK: state.topK });
  }
  return state;
}

/**
 * All lesson navigation. Pure: the model is never involved, and the same state
 * and action always give the same result.
 */
export function lessonReducer(
  state: LessonState,
  action: LessonAction,
  meta: LessonMeta,
): LessonState {
  const chapter = meta.chapters[state.chapterIndex];
  if (chapter === undefined) {
    throw new Error(`lesson: no chapter ${String(state.chapterIndex)}`);
  }
  switch (action.type) {
    case 'next': {
      if (state.stepIndex + 1 < chapter.steps.length) {
        return enterStep(state, meta, state.chapterIndex, state.stepIndex + 1);
      }
      if (state.chapterIndex + 1 < meta.chapters.length) {
        return enterStep(state, meta, state.chapterIndex + 1, 0);
      }
      return state;
    }
    case 'prev': {
      if (state.stepIndex > 0) {
        return enterStep(state, meta, state.chapterIndex, state.stepIndex - 1);
      }
      if (state.chapterIndex > 0) {
        const previous = meta.chapters[state.chapterIndex - 1];
        return enterStep(
          state,
          meta,
          state.chapterIndex - 1,
          Math.max(0, (previous?.steps.length ?? 1) - 1),
        );
      }
      return state;
    }
    case 'jumpChapter': {
      if (
        !Number.isInteger(action.chapterIndex) ||
        action.chapterIndex < 0 ||
        action.chapterIndex >= meta.chapters.length
      ) {
        return state;
      }
      return enterStep(state, meta, action.chapterIndex, 0);
    }
    case 'restart':
      return enterStep(state, meta, 0, 0);
    case 'tick': {
      if (!state.playing) return state;
      const last = lastFrame(state, meta);
      const frame = Math.min(last, state.frame + 1);
      return { ...state, frame, playing: frame < last };
    }
    case 'togglePlay': {
      if (meta.noAnimation) return state;
      const last = lastFrame(state, meta);
      if (last === 0) return state;
      if (state.frame >= last) {
        return { ...state, frame: 0, playing: true };
      }
      return { ...state, playing: !state.playing };
    }
    case 'dialUp':
      return turnDial(state, meta, -1);
    case 'dialDown':
      return turnDial(state, meta, 1);
    case 'nextHead':
    case 'prevHead': {
      if (!chapter.selectsHead || meta.nHeads <= 1) return state;
      const headIndex = wrap(state.headIndex + (action.type === 'nextHead' ? 1 : -1), meta.nHeads);
      return restartAnimation({ ...state, headIndex }, meta);
    }
    case 'cycleTopK': {
      if (!chapter.samples || meta.topKCycle.length === 0) return state;
      const at = meta.topKCycle.indexOf(state.topK);
      const topK = meta.topKCycle[(at + 1) % meta.topKCycle.length] ?? state.topK;
      return changeSampler(state, meta, { temperature: state.temperature, topK });
    }
    case 'drawn': {
      const drawn = {
        ...state,
        drawnTokenId: action.tokenId,
        drawCount: state.drawCount + 1,
      };
      // A redraw on the draw step replays the draw; a sampler change is already animating.
      return stepMeta(state, meta).restartOnDraw && state.samplerFrom === null
        ? restartAnimation(drawn, meta)
        : drawn;
    }
  }
}
