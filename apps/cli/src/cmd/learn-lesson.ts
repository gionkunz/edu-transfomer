import type { LessonHeadKv, LessonTrace, SamplingBreakdown } from '@llm-transformer-diy/inference';

import type { FlowPanelData, FlowReveal, FlowVariant } from './generate-panels';
import type { LessonState } from './learn-state';

/**
 * Building blocks of **`llm learn`**: a lesson is a list of chapters, a chapter
 * a list of steps, and a step one screen. Steps are pure: given the captured
 * forward and the current selection they return what to draw and what to say.
 * Everything numeric they show comes from `@llm-transformer-diy/inference`.
 */

/** A run of characters drawn in one style. */
export type Span = {
  readonly text: string;
  readonly color?: string;
  readonly bold?: boolean;
};

export type Line = readonly Span[];

/** What a step puts on screen for one animation frame. */
export type StepView =
  | {
      readonly kind: 'lines';
      readonly lines: readonly Line[];
      readonly explanation: string;
    }
  | {
      /** The architecture diagram of `llm generate`, drawn up to **`reveal`**. */
      readonly kind: 'flow';
      readonly data: FlowPanelData;
      readonly variant: FlowVariant;
      readonly reveal: FlowReveal;
      readonly explanation: string;
    };

/** What **`↑`/`↓`** change on a step. */
export type Dial = 'block' | 'temperature' | 'none';

/** Sizes a step's animation length may depend on. Fixed for one lesson run. */
export type LessonDims = {
  readonly nLayers: number;
  readonly promptLength: number;
};

export type LessonContext = {
  readonly trace: LessonTrace;
  /** Keys/values of the selected head, or `null` while they are being read. */
  readonly kv: LessonHeadKv | null;
  /** Display label of every prompt token, by position. */
  readonly tokenLabels: readonly string[];
  readonly labelForId: (tokenId: number) => string;
  readonly vocabSize: number;
  readonly state: LessonState;
  /** The sampler's stages for the given settings (computed in inference). */
  readonly explain: (temperature: number, topK: number) => SamplingBreakdown;
  /** Rows and columns the schematic may use. */
  readonly rows: number;
  readonly cols: number;
};

export type Step = {
  readonly id: string;
  readonly title: string;
  readonly dial: Dial;
  /** Number of animation frames; the last one is the complete picture. */
  readonly frames: (dims: LessonDims) => number;
  /** Whether drawing a new token replays this step's animation. */
  readonly restartOnDraw?: boolean;
  readonly render: (ctx: LessonContext, frame: number) => StepView;
};

export type Chapter = {
  readonly id: string;
  readonly title: string;
  /** `h`/`H` select the attention head in this chapter. */
  readonly selectsHead: boolean;
  /** `k` (top-k) and `r` (redraw) are active in this chapter. */
  readonly samples: boolean;
  readonly steps: readonly Step[];
};

/** The static shape of a lesson that navigation needs; see **`lessonReducer`**. */
export type LessonMeta = {
  readonly chapters: readonly {
    readonly selectsHead: boolean;
    readonly samples: boolean;
    readonly steps: readonly {
      readonly frames: number;
      readonly dial: Dial;
      readonly restartOnDraw: boolean;
    }[];
  }[];
  readonly nLayers: number;
  readonly nHeads: number;
  /** Top-k settings **`k`** cycles through; `0` means off. */
  readonly topKCycle: readonly number[];
  readonly noAnimation: boolean;
};

/** Top-k settings offered by **`k`**: off, a few common values, and the command-line one. */
export function topKCycleFor(cliTopK: number): number[] {
  const values = [0, 5, 10, 40];
  if (Number.isInteger(cliTopK) && cliTopK > 0 && !values.includes(cliTopK)) {
    values.push(cliTopK);
  }
  return [0, ...values.filter((v) => v > 0).sort((a, b) => a - b)];
}

export function buildLessonMeta(
  chapters: readonly Chapter[],
  dims: LessonDims,
  options: {
    readonly nHeads: number;
    readonly cliTopK: number;
    readonly noAnimation: boolean;
  },
): LessonMeta {
  return {
    chapters: chapters.map((c) => ({
      selectsHead: c.selectsHead,
      samples: c.samples,
      steps: c.steps.map((s) => ({
        frames: Math.max(1, s.frames(dims)),
        dial: s.dial,
        restartOnDraw: s.restartOnDraw === true,
      })),
    })),
    nLayers: dims.nLayers,
    nHeads: options.nHeads,
    topKCycle: topKCycleFor(options.cliTopK),
    noAnimation: options.noAnimation,
  };
}

export function totalSteps(meta: LessonMeta): number {
  return meta.chapters.reduce((sum, c) => sum + c.steps.length, 0);
}

/** 0-based position of a step among all steps of the lesson. */
export function overallStepIndex(
  meta: LessonMeta,
  chapterIndex: number,
  stepIndex: number,
): number {
  let n = 0;
  for (let c = 0; c < chapterIndex; c++) {
    n += meta.chapters[c]?.steps.length ?? 0;
  }
  return n + stepIndex;
}

export function isLastStep(meta: LessonMeta, chapterIndex: number, stepIndex: number): boolean {
  return overallStepIndex(meta, chapterIndex, stepIndex) === totalSteps(meta) - 1;
}
