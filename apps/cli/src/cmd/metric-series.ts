/** One x-position of a downsampled metric curve: all samples whose step falls in the bucket. */
export type MetricBucket = {
  /** First training step (1-based, completed steps) that landed in this bucket. */
  readonly step: number;
  readonly mean: number;
  readonly min: number;
  readonly max: number;
  readonly count: number;
};

type MutableBucket = {
  slot: number;
  step: number;
  sum: number;
  min: number;
  max: number;
  count: number;
};

/**
 * Fixed-memory history of a per-step metric for charting a whole run.
 *
 * Samples are grouped into buckets of **`stride`** steps. When the bucket count
 * exceeds **`maxBuckets`**, the stride doubles and neighbours merge, so a
 * 500-step run and a 500k-step run both end up with a few hundred points and
 * the chart always spans the full run. Non-finite samples are dropped.
 */
export class MetricSeries {
  private readonly maxBuckets: number;
  private stride = 1;
  private data: MutableBucket[] = [];

  constructor(maxBuckets = 240) {
    if (!Number.isInteger(maxBuckets) || maxBuckets < 2) {
      throw new Error('MetricSeries: maxBuckets must be an integer ≥ 2');
    }
    this.maxBuckets = maxBuckets;
  }

  push(step: number, value: number): void {
    if (!Number.isFinite(value) || !Number.isFinite(step)) return;
    const slot = Math.floor(step / this.stride);
    const last = this.data[this.data.length - 1];
    if (last !== undefined && last.slot === slot) {
      last.sum += value;
      last.count += 1;
      last.min = Math.min(last.min, value);
      last.max = Math.max(last.max, value);
      return;
    }
    this.data.push({ slot, step, sum: value, min: value, max: value, count: 1 });
    while (this.data.length > this.maxBuckets) {
      this.halve();
    }
  }

  private halve(): void {
    this.stride *= 2;
    const merged: MutableBucket[] = [];
    for (const b of this.data) {
      const slot = Math.floor(b.step / this.stride);
      const last = merged[merged.length - 1];
      if (last !== undefined && last.slot === slot) {
        last.sum += b.sum;
        last.count += b.count;
        last.min = Math.min(last.min, b.min);
        last.max = Math.max(last.max, b.max);
      } else {
        merged.push({ ...b, slot });
      }
    }
    this.data = merged;
  }

  get length(): number {
    return this.data.length;
  }

  snapshot(): MetricBucket[] {
    return this.data.map((b) => ({
      step: b.step,
      mean: b.sum / b.count,
      min: b.min,
      max: b.max,
      count: b.count,
    }));
  }
}
