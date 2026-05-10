/** Cosine decay with linear warmup (matches `docs/project.md`). */
export type CosineLrSchedule = {
  readonly lrMax: number;
  readonly lrMin: number;
  readonly warmupSteps: number;
  readonly totalSteps: number;
};

/**
 * Learning rate at **`step`** (0-based training step index).
 * Warmup: **`lr_max * (step + 1) / warmup_steps`** for **`step < warmupSteps`** (first step is non-zero).
 * Cosine: same formula with **progress** in **`[0, 1]`** over post-warmup steps (last training step maps to **`progress === 1`**).
 */
export function getCosineLr(step: number, schedule: CosineLrSchedule): number {
  if (!(schedule.warmupSteps >= 0) || !Number.isInteger(schedule.warmupSteps)) {
    throw new Error('getCosineLr: warmupSteps must be a non-negative integer');
  }
  if (!(schedule.totalSteps > 0) || !Number.isInteger(schedule.totalSteps)) {
    throw new Error('getCosineLr: totalSteps must be a positive integer');
  }
  if (!(schedule.lrMax >= 0) || !(schedule.lrMin >= 0)) {
    throw new Error('getCosineLr: lrMin and lrMax must be non-negative');
  }
  const { lrMax, lrMin, warmupSteps, totalSteps } = schedule;

  if (step < warmupSteps) {
    const denom = Math.max(1, warmupSteps);
    return lrMax * ((step + 1) / denom);
  }

  const postLen = totalSteps - warmupSteps;
  if (postLen <= 0) {
    return lrMin;
  }
  const t = step - warmupSteps;
  const progress = postLen <= 1 ? 1 : Math.min(1, Math.max(0, t / (postLen - 1)));
  return lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * progress));
}
