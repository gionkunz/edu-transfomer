import { CELL_WIDTH, legend, lineChart, type ColorMode } from '@chartist-js/terminal';
import { getCosineLr, type CosineLrSchedule } from '@llm-transformer-diy/training';

import type { MetricBucket } from './metric-series';
import type { ValLossPoint } from './train-dash-types';

/**
 * Chart strings for the **`llm train`** dashboard, rendered by
 * `@chartist-js/terminal` (braille line charts). Pure functions of the metric
 * history: no terminal access, so they are unit-testable and reusable for a
 * post-run report.
 */

export type ChartSize = {
  readonly columns: number;
  readonly rows: number;
  readonly color: ColorMode;
};

/** Train / validation / reference colors; picked to stay distinct on dark and light terminals. */
const PALETTE_LOSS = ['#4aa3ff', '#ffb000', '#8a8a8a'] as const;
const PALETTE_LR = ['#3ecf8e', '#6b7280'] as const;
const PALETTE_GRAD = ['#c678dd', '#e06c75'] as const;

/** One x-axis label per this many columns; the rest are blank so labels never collide. */
const COLUMNS_PER_X_LABEL = 16;

function formatStep(step: number): string {
  if (step >= 1_000_000) return `${(step / 1_000_000).toFixed(1)}M`;
  if (step >= 10_000) return `${String(Math.round(step / 1000))}k`;
  if (step >= 1000) return `${(step / 1000).toFixed(1)}k`;
  return String(step);
}

function sparseStepLabels(steps: readonly number[], columns: number): string[] {
  const target = Math.max(2, Math.floor(columns / COLUMNS_PER_X_LABEL));
  const every = Math.max(1, Math.ceil(steps.length / target));
  return steps.map((s, i) => (i % every === 0 ? formatStep(s) : ''));
}

function round(value: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}

/** Index of the bucket whose step is closest to **`step`** (buckets are step-ascending). */
function nearestBucketIndex(buckets: readonly MetricBucket[], step: number): number {
  let best = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let i = 0; i < buckets.length; i++) {
    const b = buckets[i];
    if (b === undefined) continue;
    const d = Math.abs(b.step - step);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

const cartesian = {
  showPoint: false,
  fullWidth: true,
  axisX: { showGrid: false },
} as const;

/**
 * Train loss (bucket means) with the validation loss overlaid.
 *
 * The early part of a run falls from `ln(V)` by several nats and would flatten
 * everything after it, so the y-range is clamped to the 98th percentile of the
 * train curve: the first buckets clip at the top edge and the interesting part
 * of the curve keeps its resolution.
 */
export function renderLossChart(
  lossHistory: readonly MetricBucket[],
  valHistory: readonly ValLossPoint[],
  size: ChartSize,
): string | null {
  if (lossHistory.length < 2) return null;
  const train = lossHistory.map((b) => round(b.mean, 3));
  const sorted = [...train].sort((a, b) => a - b);
  const p98 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))] ?? 0;
  const lowest = sorted[0] ?? 0;
  const valValues = valHistory.map((v) => v.loss).filter((v) => Number.isFinite(v));
  const high = round(Math.max(p98, ...valValues.slice(-Math.max(1, valValues.length - 1))), 2);
  const low = round(Math.min(lowest, ...valValues), 2);

  // Validation is measured every N steps, the train curve every bucket. Chartist
  // draws a line only through consecutive values, so interpolate linearly between
  // measurements to get one unbroken run from the first to the last of them.
  const val: (number | null)[] = train.map(() => null);
  const anchors = valHistory
    .filter((v) => Number.isFinite(v.loss))
    .map((v) => ({ index: nearestBucketIndex(lossHistory, v.step), loss: v.loss }));
  for (let a = 0; a < anchors.length; a++) {
    const from = anchors[a];
    const to = anchors[a + 1] ?? from;
    if (from === undefined || to === undefined) continue;
    const span = Math.max(1, to.index - from.index);
    for (let i = from.index; i <= to.index; i++) {
      val[i] = round(from.loss + ((to.loss - from.loss) * (i - from.index)) / span, 3);
    }
  }
  const hasVal = val.filter((v) => v !== null).length >= 2;

  const series = [{ name: 'train', data: train as (number | null)[] }];
  if (hasVal) series.push({ name: 'validation', data: val });
  const data = {
    labels: sparseStepLabels(
      lossHistory.map((b) => b.step),
      size.columns,
    ),
    series,
  };
  const palette = [...PALETTE_LOSS];
  const chart = lineChart(data, {
    ...cartesian,
    columns: size.columns,
    rows: size.rows,
    color: size.color,
    palette,
    ...(high > low ? { high, low } : {}),
  });
  return `${chart}\n${legend(data, { columns: size.columns, color: size.color, palette })}`;
}

/** The whole LR schedule (warmup + cosine): the part already trained in color, the rest in gray. */
export function renderLrChart(
  schedule: CosineLrSchedule,
  stepsCompleted: number,
  size: ChartSize,
): string {
  const points = Math.max(2, Math.min(120, size.columns * 2, schedule.totalSteps));
  const steps: number[] = [];
  const remaining: (number | null)[] = [];
  const done: (number | null)[] = [];
  let boundaryPlaced = false;
  for (let i = 0; i < points; i++) {
    const step = Math.round((i * (schedule.totalSteps - 1)) / (points - 1));
    const lr = getCosineLr(step, schedule);
    steps.push(step);
    const isDone = step < stepsCompleted;
    done.push(isDone ? lr : null);
    // The first not-yet-trained point belongs to both runs so the two lines join.
    if (!isDone && !boundaryPlaced && i > 0) {
      done[i] = lr;
      boundaryPlaced = true;
    }
    remaining.push(isDone ? null : lr);
  }
  const data = {
    labels: sparseStepLabels(steps, size.columns),
    series: [
      { name: 'done', data: done },
      { name: 'remaining', data: remaining },
    ],
  };
  const formatLr = (v: unknown): string => Number(v).toExponential(0);
  return lineChart(data, {
    ...cartesian,
    columns: size.columns,
    rows: size.rows,
    color: size.color,
    palette: [...PALETTE_LR],
    low: 0,
    axisY: { labelInterpolationFnc: formatLr, offset: 6 * CELL_WIDTH },
  });
}

/** Pre-clip gradient norm over the run, with the clip threshold as a flat reference line. */
export function renderGradNormChart(
  gradNormHistory: readonly MetricBucket[],
  maxGradNorm: number | null,
  size: ChartSize,
): string | null {
  if (gradNormHistory.length < 2) return null;
  const norms = gradNormHistory.map((b) => round(b.mean, 3));
  const sorted = [...norms].sort((a, b) => a - b);
  const p98 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))] ?? 1;
  const series = [{ name: 'grad norm', data: norms as (number | null)[] }];
  if (maxGradNorm !== null) {
    series.push({ name: 'clip', data: norms.map(() => maxGradNorm) });
  }
  const high = round(Math.max(p98, maxGradNorm ?? 0) * 1.1, 2);
  return lineChart(
    {
      labels: sparseStepLabels(
        gradNormHistory.map((b) => b.step),
        size.columns,
      ),
      series,
    },
    {
      ...cartesian,
      columns: size.columns,
      rows: size.rows,
      color: size.color,
      palette: [...PALETTE_GRAD],
      low: 0,
      ...(high > 0 ? { high } : {}),
    },
  );
}
