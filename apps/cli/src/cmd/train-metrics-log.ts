import { createWriteStream, type WriteStream } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';

/** File name of the per-step metrics log inside a checkpoint directory. */
export const TRAIN_METRICS_FILE = 'metrics.jsonl';

/**
 * One line of **`metrics.jsonl`**: the state after **`step`** completed optimizer steps.
 * **`valLoss`** is present only on steps where validation ran.
 */
export type TrainMetricsRecord = {
  readonly step: number;
  readonly loss: number;
  readonly lr: number;
  readonly gradNorm: number | null;
  readonly tokPerSec: number;
  readonly valLoss?: number;
};

function parseRecord(line: string): TrainMetricsRecord | null {
  if (line.trim().length === 0) return null;
  try {
    const o = JSON.parse(line) as Record<string, unknown>;
    if (typeof o.step !== 'number' || typeof o.loss !== 'number' || typeof o.lr !== 'number') {
      return null;
    }
    return {
      step: o.step,
      loss: o.loss,
      lr: o.lr,
      gradNorm: typeof o.gradNorm === 'number' ? o.gradNorm : null,
      tokPerSec: typeof o.tokPerSec === 'number' ? o.tokPerSec : 0,
      ...(typeof o.valLoss === 'number' ? { valLoss: o.valLoss } : {}),
    };
  } catch {
    // A run killed mid-write can leave a torn last line; skip it.
    return null;
  }
}

/** Reads every well-formed record; a missing file is an empty history. */
export async function readTrainMetrics(filePath: string): Promise<TrainMetricsRecord[]> {
  let text: string;
  try {
    text = await readFile(filePath, 'utf8');
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw err;
  }
  const out: TrainMetricsRecord[] = [];
  for (const line of text.split('\n')) {
    const rec = parseRecord(line);
    if (rec !== null) out.push(rec);
  }
  return out;
}

/**
 * Append-only writer for **`metrics.jsonl`**.
 *
 * **`keep`** is the history to preserve: empty for a fresh run, or the records up
 * to the resume step for **`--resume`** (anything logged after the last
 * checkpoint is dropped, because those steps will be trained again).
 */
export class TrainMetricsLog {
  private readonly stream: WriteStream;

  private constructor(stream: WriteStream) {
    this.stream = stream;
  }

  static async open(
    filePath: string,
    keep: readonly TrainMetricsRecord[],
  ): Promise<TrainMetricsLog> {
    const head = keep.map((r) => `${JSON.stringify(r)}\n`).join('');
    await writeFile(filePath, head, 'utf8');
    return new TrainMetricsLog(createWriteStream(filePath, { flags: 'a' }));
  }

  write(record: TrainMetricsRecord): void {
    this.stream.write(`${JSON.stringify(record)}\n`);
  }

  close(): Promise<void> {
    return new Promise((resolve) => {
      this.stream.end(resolve);
    });
  }
}
