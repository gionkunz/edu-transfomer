import { describe, expect, it } from 'vitest';

import {
  resolveCheckpointEvery,
  shouldSavePeriodicCheckpoint,
  TRAIN_DEFAULT_CHECKPOINT_EVERY,
} from './train-checkpoint';

describe('resolveCheckpointEvery', () => {
  it('defaults to 2000 for long runs', () => {
    expect(resolveCheckpointEvery(50_000, TRAIN_DEFAULT_CHECKPOINT_EVERY)).toBe(2000);
  });

  it('allows 0 to disable mid-run saves', () => {
    expect(resolveCheckpointEvery(100, 0)).toBe(0);
  });

  it('rejects negative intervals', () => {
    expect(() => resolveCheckpointEvery(100, -1)).toThrow(/non-negative integer/);
  });
});

describe('shouldSavePeriodicCheckpoint', () => {
  it('fires on multiples of checkpointEvery before totalSteps', () => {
    expect(shouldSavePeriodicCheckpoint(2000, 50_000, 2000)).toBe(true);
    expect(shouldSavePeriodicCheckpoint(1999, 50_000, 2000)).toBe(false);
  });

  it('does not fire at totalSteps (final save handles that)', () => {
    expect(shouldSavePeriodicCheckpoint(50_000, 50_000, 2000)).toBe(false);
  });

  it('never fires when checkpointEvery is 0', () => {
    expect(shouldSavePeriodicCheckpoint(10_000, 50_000, 0)).toBe(false);
  });
});
