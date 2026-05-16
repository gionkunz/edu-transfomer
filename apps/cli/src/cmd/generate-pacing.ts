/** ms between tokens in slow mode (0.25 tok/s). */
export const PACING_SLOW_DELAY_MS = 4000;

export type SpeedMode = 'max' | 'slow';

/**
 * Host-side pacing for the generate explore TUI: pluggable into the
 * **`generateAutoregressive`** **`onStep`** await point so the inference loop
 * stays free of UI/timing logic.
 *
 * State machine:
 * - **`speed`** — **`'max'`** (no wait) or **`'slow'`** (≈ 4 s sleep between tokens).
 * - **`paused`** — when **`true`**, **`waitBetweenSteps()`** blocks indefinitely on a gate
 *   that resolves on the next **`togglePause()`** or on **`cancel()`**.
 *
 * Both the slow timer and the pause gate are released by **`cancel()`** so teardown
 * (e.g. **`q`** / **`Ctrl+C`**) never deadlocks.
 */
export class PacingController {
  private _speed: SpeedMode = 'max';
  private _paused = false;
  private cancelled = false;
  private slowTimer: ReturnType<typeof setTimeout> | null = null;
  private slowResolve: (() => void) | null = null;
  private pauseResolvers: Array<() => void> = [];
  private readonly slowDelayMs: number;

  constructor(opts?: { readonly slowDelayMs?: number }) {
    this.slowDelayMs = opts?.slowDelayMs ?? PACING_SLOW_DELAY_MS;
  }

  get speed(): SpeedMode {
    return this._speed;
  }

  get paused(): boolean {
    return this._paused;
  }

  /** UI-friendly status string; **`PAUSED`** wins over **`SLOW`** when both apply. */
  get statusLabel(): 'NORMAL' | 'SLOW' | 'PAUSED' {
    if (this._paused) return 'PAUSED';
    if (this._speed === 'slow') return 'SLOW';
    return 'NORMAL';
  }

  setSpeed(s: SpeedMode): void {
    this._speed = s;
    if (s === 'max') {
      this.flushSlowTimer();
    }
  }

  toggleSpeed(): void {
    this.setSpeed(this._speed === 'slow' ? 'max' : 'slow');
  }

  togglePause(): void {
    this._paused = !this._paused;
    if (!this._paused) {
      this.flushPauseGate();
    }
  }

  /** Resolve any in-flight wait so callers can tear down without hanging. */
  cancel(): void {
    this.cancelled = true;
    this.flushSlowTimer();
    this.flushPauseGate();
  }

  /**
   * Awaitable host hook for **`generateAutoregressive`**'s **`onStep`**.
   * Sleeps ≈ 4 s in **`slow`** mode, then awaits the pause gate while paused.
   * Returns immediately if **`cancel()`** has been called.
   */
  async waitBetweenSteps(): Promise<void> {
    if (this.cancelled) return;
    if (this._speed === 'slow') {
      await new Promise<void>((resolve) => {
        this.slowResolve = resolve;
        this.slowTimer = setTimeout(() => {
          this.slowTimer = null;
          this.slowResolve = null;
          resolve();
        }, this.slowDelayMs);
      });
    }
    if (this.cancelled) return;
    while (this._paused && !this.cancelled) {
      await new Promise<void>((resolve) => {
        this.pauseResolvers.push(resolve);
      });
    }
  }

  private flushSlowTimer(): void {
    if (this.slowTimer !== null) {
      clearTimeout(this.slowTimer);
      this.slowTimer = null;
    }
    if (this.slowResolve !== null) {
      const r = this.slowResolve;
      this.slowResolve = null;
      r();
    }
  }

  private flushPauseGate(): void {
    const r = this.pauseResolvers.splice(0);
    for (const fn of r) fn();
  }
}
