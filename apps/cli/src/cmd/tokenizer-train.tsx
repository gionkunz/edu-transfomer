import React from 'react';
import { Box, Text, render, useInput } from '@wolf-tui/react';
import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';

import { discoverTrainingFiles } from '@llm-transformer-diy/io';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';

import { primeTuiTerminalCanvas } from '../tui-prime';

/** Default GPT-2-ish specials (**`floor = S + 256`**) merge **upper bound** for progress display only. */
const DEFAULT_MERGE_LOWER_FLOOR_IDS = 2 + 256;

export type RunTokenizerTrainOpts = {
  readonly dataGlobs: string[];
  readonly vocabSize: number;
  readonly outPath: string;
  /**
   * Stream via on-disk **u32** token file (**`BPETokenizer.trainFromFiles`**).
   * When **`false`** (default), load corpus into RAM and call **`BPETokenizer.train`** (faster for data that fits).
   */
  readonly diskStream?: boolean;
};

type TrainVm = {
  mode: 'ram' | 'disk';
  phase: 'load' | 'train' | 'materialize' | 'merge' | 'save' | 'done' | 'error';
  fileCount: number;
  corpusUtf8MiB: number;
  /** Disk streaming only. */
  tokenStreamFileMiB: number;
  mergesDone: number;
  mergesCapApprox: number;
  /** In-memory **`Uint32`** token buffer size (**`MiB`**) during RAM merges; disk stream size during **`merge`**. */
  streamMiB: number;
  vocabSize: number;
  finalSize: number;
  errLine: string;
};

function bigToApproxMiB(n: bigint): number {
  return Number(n) / (1024 * 1024);
}

function utf8BytesToApproxMiB(bytes: Uint8Array): number {
  return bytes.byteLength / (1024 * 1024);
}

function TokTrainView(props: {
  vm: TrainVm;
  /** Unmount renderer and **`process.exit(exitCode)`** — training cannot be cancelled mid-merge. */
  onExit: (exitCode: number) => void;
}) {
  const { vm, onExit } = props;
  useInput((input, key) => {
    if (input === 'q' || input === 'Q') {
      onExit(0);
    }
    if (key.escape) {
      onExit(0);
    }
    if (key.ctrl && input === 'c') {
      onExit(130);
    }
    if (key.ctrl && input === 'x') {
      onExit(0);
    }
  });

  const modeHint = vm.mode === 'disk' ? 'disk stream' : 'RAM';

  let phaseLabel: string = vm.phase;
  if (vm.phase === 'load') phaseLabel = 'Reading corpus files into RAM';
  if (vm.phase === 'train') phaseLabel = 'BPE training in RAM · starting merges';
  if (vm.phase === 'materialize') phaseLabel = 'Materialize corpus → disk token stream (u32 LE)';
  if (vm.phase === 'merge') {
    phaseLabel =
      vm.mode === 'ram' ? 'BPE merge passes (in-memory)' : 'BPE merge passes (in-place merge)';
  }
  if (vm.phase === 'save') phaseLabel = 'Saving tokenizer JSON';
  if (vm.phase === 'done') phaseLabel = 'Done';

  const mergePendingLoadOrMaterialize =
    vm.phase === 'load' || (vm.mode === 'disk' && vm.phase === 'materialize');
  const mergePendingRamBeforeFirst =
    vm.mode === 'ram' && vm.phase === 'train' && vm.mergesDone === 0;

  const mergeLine =
    mergePendingLoadOrMaterialize || mergePendingRamBeforeFirst
      ? `merges (pending) · cap ~${String(vm.mergesCapApprox)}`
      : `merges ${String(vm.mergesDone)} · cap ~${String(vm.mergesCapApprox)} · stream ~${vm.streamMiB.toFixed(
          2,
        )} MiB`;

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', padding: 1, gap: 0 }}>
      <Text style={{ fontWeight: 'bold' }}>{`Tokenizer train · BPE (${modeHint})`}</Text>
      <Text style={{ color: 'gray' }}>{`files ${String(vm.fileCount)} · vocab target ${String(
        vm.vocabSize,
      )} · quit: q / Esc / Ctrl+C / Ctrl+X (yields often; one merge scan can still stall briefly on GB-scale streams)`}</Text>
      <Text>{`phase · ${phaseLabel}`}</Text>
      <Text>{`corpus UTF-8 ~${vm.corpusUtf8MiB.toFixed(2)} MiB (file-separator newlines not counted)`}</Text>
      {vm.mode === 'disk' && vm.phase === 'materialize' ? (
        <Text
          style={{ color: 'gray' }}
        >{`disk token stream written ~${vm.tokenStreamFileMiB.toFixed(
          2,
        )} MiB (expect up to ~4x corpus UTF-8)`}</Text>
      ) : null}
      <Text>{mergeLine}</Text>
      {vm.phase === 'done' ? (
        <Text style={{ color: 'cyan' }}>{`saved vocab ids ${String(vm.finalSize)}`}</Text>
      ) : null}
      {vm.phase === 'error' ? <Text style={{ color: 'red' }}>{vm.errLine}</Text> : null}
    </Box>
  );
}

/** Join training files like the disk path (**`\n`** between UTF‑8 payloads) **without** materializing one giant **`string`**. Copies into the combined buffer **in stripes** (**`setImmediate`**) so **`SIGINT`/quit keys** can interrupt huge **`Buffer.concat`**-equivalent work. */
async function concatCorpusAsUtf8Bytes(
  absPaths: readonly string[],
  opts?: {
    readonly onAccumulatedUtf8Bytes?: (n: bigint) => void;
  },
): Promise<Uint8Array> {
  const sepByte = Buffer.from('\n', 'utf8');
  let total = BigInt(0);
  const chunks: Buffer[] = [];
  for (let i = 0; i < absPaths.length; i++) {
    const p = absPaths[i];
    if (p === undefined) continue;
    if (i > 0) {
      chunks.push(sepByte);
      total += BigInt(sepByte.length);
    }
    const buf = await readFile(p);
    chunks.push(buf as Buffer);
    total += BigInt(buf.length);
    opts?.onAccumulatedUtf8Bytes?.(total);
  }
  const maxSafe = BigInt(Number.MAX_SAFE_INTEGER);
  if (total > maxSafe) {
    throw new Error(
      'tokenizer train: combined corpus exceeds safe integer size — use **--disk-stream** or split the input',
    );
  }
  const totalN = Number(total);
  if (totalN === 0) {
    return Buffer.alloc(0);
  }
  const out = Buffer.allocUnsafe(totalN);
  const yieldEveryBytes = 32 * 1024 * 1024;
  let sinceYield = 0;
  let offset = 0;
  for (const chunk of chunks) {
    let src = 0;
    while (src < chunk.length) {
      const take = Math.min(chunk.length - src, yieldEveryBytes - sinceYield);
      chunk.copy(out, offset, src, src + take);
      offset += take;
      src += take;
      sinceYield += take;
      if (sinceYield >= yieldEveryBytes) {
        sinceYield = 0;
        await new Promise<void>((r) => {
          globalThis.setImmediate(r);
        });
      }
    }
  }
  return out;
}

function ramTrainFailureHint(e: unknown): Error {
  const msg = e instanceof Error ? e.message : String(e);
  const lower = msg.toLowerCase();
  if (
    msg.includes('Invalid string length') ||
    msg.includes('invalid array length') ||
    lower.includes('heap') ||
    lower.includes('allocation failed') ||
    lower.includes('out of memory')
  ) {
    return new Error(
      `${msg}\ntokenizer train: corpus may exceed the Node/V8 usable heap — try **--disk-stream**, or increase heap (**NODE_OPTIONS=\"--max-old-space-size=...\"**) — RAM BPE peaks around **~(5–9)×** corpus UTF‑8 (**u32 token stream**, merge buffers, bookkeeping)`,
    );
  }
  return e instanceof Error ? e : new Error(msg);
}

async function tokenizerTrainPlain(opts: RunTokenizerTrainOpts): Promise<void> {
  const files = await discoverTrainingFiles(opts.dataGlobs);
  if (files.length === 0) {
    throw new Error('tokenizer train: no files matched --data');
  }
  const abs = files.map((rel) => path.resolve(rel));
  const mergesCapApprox = Math.max(0, opts.vocabSize - DEFAULT_MERGE_LOWER_FLOOR_IDS);
  const diskStream = opts.diskStream === true;

  const tok = new BPETokenizer();

  if (diskStream) {
    console.error(
      `Training BPE (disk-stream): ${String(files.length)} file(s); ~${String(mergesCapApprox)} merges max`,
    );
    await tok.trainFromFiles(abs, opts.vocabSize, {
      hooks: {
        onCorpusUtf8Bytes(b) {
          console.error(`  corpus UTF-8 read ~${bigToApproxMiB(b).toFixed(2)} MiB`);
        },
        onTokenStreamBytesWritten(nb) {
          console.error(`  token stream written ~${bigToApproxMiB(nb).toFixed(2)} MiB`);
        },
        onMergeComplete(ev) {
          const idx = ev.mergeIndex + 1;
          if (idx % 32 === 0 || idx <= 4) {
            console.error(
              `  merge ${String(idx)} / ~${String(mergesCapApprox)} · stream ${bigToApproxMiB(ev.tokenStreamBytes).toFixed(1)} MiB`,
            );
          }
        },
      },
    });
  } else {
    console.error(
      `Training BPE (RAM): ${String(files.length)} file(s); ~${String(mergesCapApprox)} merges max`,
    );
    let corpusBytes: Uint8Array;
    try {
      corpusBytes = await concatCorpusAsUtf8Bytes(abs);
    } catch (e) {
      throw ramTrainFailureHint(e);
    }
    console.error(`  corpus UTF-8 ~${utf8BytesToApproxMiB(corpusBytes).toFixed(2)} MiB`);
    try {
      await tok.trainAsyncOnUtf8Bytes(corpusBytes, opts.vocabSize, {
        hooks: {
          onMergeComplete(ev) {
            const idx = ev.mergeIndex + 1;
            if (idx % 32 === 0 || idx <= 4) {
              console.error(
                `  merge ${String(idx)} / ~${String(mergesCapApprox)} · in-memory stream ${bigToApproxMiB(ev.tokenStreamBytes).toFixed(1)} MiB`,
              );
            }
          },
          yieldToHost: async () =>
            await new Promise<void>((r) => {
              globalThis.setImmediate(() => r());
            }),
        },
      });
    } catch (e) {
      throw ramTrainFailureHint(e);
    }
  }

  await mkdir(path.dirname(path.resolve(opts.outPath)), { recursive: true });
  await tok.save(opts.outPath);
  console.error(`Wrote tokenizer: ${opts.outPath}`);
}

function useRichTokenizerTui(): boolean {
  try {
    return Boolean(process.stderr.isTTY) && process.env.TERM !== 'dumb';
  } catch {
    return false;
  }
}

async function tokenizerTrainRamTui(opts: RunTokenizerTrainOpts): Promise<void> {
  const files = await discoverTrainingFiles(opts.dataGlobs);
  if (files.length === 0) {
    throw new Error('tokenizer train: no files matched --data');
  }
  const abs = files.map((rel) => path.resolve(rel));
  const mergesCapApprox = Math.max(0, opts.vocabSize - DEFAULT_MERGE_LOWER_FLOOR_IDS);

  const vm: TrainVm = {
    mode: 'ram',
    phase: 'load',
    fileCount: files.length,
    corpusUtf8MiB: 0,
    tokenStreamFileMiB: 0,
    mergesDone: 0,
    mergesCapApprox,
    streamMiB: 0,
    vocabSize: opts.vocabSize,
    finalSize: 0,
    errLine: '',
  };

  primeTuiTerminalCanvas();
  let instRef: ReturnType<typeof render> | null = null;

  let exiting = false;
  function exitTokenizerTrainTui(code: number): void {
    if (exiting) {
      return;
    }
    exiting = true;
    try {
      instRef?.unmount();
    } catch {
      /* ignore */
    }
    instRef = null;
    process.exit(code);
  }

  function paint(): void {
    // Shallow-clone `vm` so React reconciles (we mutate one object in place).
    instRef?.rerender(<TokTrainView vm={{ ...vm }} onExit={exitTokenizerTrainTui} />);
  }

  const onSigInt = (): void => {
    exitTokenizerTrainTui(130);
  };
  process.on('SIGINT', onSigInt);

  instRef = render(<TokTrainView vm={{ ...vm }} onExit={exitTokenizerTrainTui} />, {
    exitOnCtrlC: false,
    patchConsole: true,
  });

  const tok = new BPETokenizer();

  try {
    const corpusBytes = await concatCorpusAsUtf8Bytes(abs, {
      onAccumulatedUtf8Bytes(n) {
        vm.corpusUtf8MiB = bigToApproxMiB(n);
        paint();
      },
    });

    vm.mergesDone = 0;
    vm.phase = 'train';
    paint();
    await new Promise<void>((r) => {
      globalThis.setImmediate(r);
    });

    try {
      await tok.trainAsyncOnUtf8Bytes(corpusBytes, opts.vocabSize, {
        hooks: {
          onPairIndexBuilt(_distinctInitialPairs) {
            // No new TUI element yet — surfaced through the train log only when
            // the plain (non-TUI) path is in use; here we just keep the hook
            // wired so future visualizations can consume it without a perf hit.
          },
          onMergeComplete(ev) {
            vm.phase = 'merge';
            vm.mergesDone = ev.mergeIndex + 1;
            vm.streamMiB = bigToApproxMiB(ev.tokenStreamBytes);
            paint();
          },
          yieldToHost: async () =>
            await new Promise<void>((r) => {
              globalThis.setImmediate(() => r());
            }),
        },
      });
    } catch (e) {
      throw ramTrainFailureHint(e);
    }

    vm.phase = 'save';
    paint();
    await mkdir(path.dirname(path.resolve(opts.outPath)), { recursive: true });
    await tok.save(opts.outPath);
    vm.phase = 'done';
    vm.finalSize = tok.vocabSize;
    paint();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 400);
    });
  } catch (e) {
    vm.phase = 'error';
    vm.errLine = e instanceof Error ? e.message : String(e);
    paint();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 900);
    });
    throw e;
  } finally {
    process.off('SIGINT', onSigInt);
    if (instRef) {
      instRef.unmount();
      instRef = null;
    }
    if (!exiting && vm.phase !== 'error') {
      console.error(`Wrote tokenizer: ${opts.outPath}`);
    }
  }
}

async function tokenizerTrainDiskStreamTui(opts: RunTokenizerTrainOpts): Promise<void> {
  const files = await discoverTrainingFiles(opts.dataGlobs);
  if (files.length === 0) {
    throw new Error('tokenizer train: no files matched --data');
  }
  const abs = files.map((rel) => path.resolve(rel));
  const mergesCapApprox = Math.max(0, opts.vocabSize - DEFAULT_MERGE_LOWER_FLOOR_IDS);

  const vm: TrainVm = {
    mode: 'disk',
    phase: 'materialize',
    fileCount: files.length,
    corpusUtf8MiB: 0,
    tokenStreamFileMiB: 0,
    mergesDone: 0,
    mergesCapApprox,
    streamMiB: 0,
    vocabSize: opts.vocabSize,
    finalSize: 0,
    errLine: '',
  };

  primeTuiTerminalCanvas();
  let instRef: ReturnType<typeof render> | null = null;

  let exiting = false;
  function exitTokenizerTrainTui(code: number): void {
    if (exiting) {
      return;
    }
    exiting = true;
    try {
      instRef?.unmount();
    } catch {
      /* ignore */
    }
    instRef = null;
    process.exit(code);
  }

  function paint(): void {
    // Shallow-clone `vm` so React reconciles (we mutate one object in place).
    instRef?.rerender(<TokTrainView vm={{ ...vm }} onExit={exitTokenizerTrainTui} />);
  }

  const onSigInt = (): void => {
    exitTokenizerTrainTui(130);
  };
  process.on('SIGINT', onSigInt);

  instRef = render(<TokTrainView vm={{ ...vm }} onExit={exitTokenizerTrainTui} />, {
    exitOnCtrlC: false,
    patchConsole: true,
  });

  const tok = new BPETokenizer();
  try {
    await tok.trainFromFiles(abs, opts.vocabSize, {
      hooks: {
        onCorpusUtf8Bytes(b) {
          vm.phase = 'materialize';
          vm.corpusUtf8MiB = bigToApproxMiB(b);
          paint();
        },
        onTokenStreamBytesWritten(nb) {
          // Single end-of-materialize emit under the incremental path: the
          // value reflects the slot pool size in u32-equivalent units and stays
          // constant for the rest of the run (no per-merge file rewrite anymore).
          vm.phase = 'materialize';
          vm.tokenStreamFileMiB = bigToApproxMiB(nb);
          paint();
        },
        onPairIndexBuilt(_distinctInitialPairs) {
          // Wired for future TUI consumption; no display element yet.
        },
        onMergeComplete(ev) {
          vm.phase = 'merge';
          vm.mergesDone = ev.mergeIndex + 1;
          vm.streamMiB = bigToApproxMiB(ev.tokenStreamBytes);
          paint();
        },
        yieldToHost: async () =>
          await new Promise<void>((r) => {
            globalThis.setImmediate(() => r());
          }),
      },
    });

    vm.phase = 'save';
    paint();
    await mkdir(path.dirname(path.resolve(opts.outPath)), { recursive: true });
    await tok.save(opts.outPath);
    vm.phase = 'done';
    vm.finalSize = tok.vocabSize;
    paint();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 400);
    });
  } catch (e) {
    vm.phase = 'error';
    vm.errLine = e instanceof Error ? e.message : String(e);
    paint();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 900);
    });
    throw e;
  } finally {
    process.off('SIGINT', onSigInt);
    if (instRef) {
      instRef.unmount();
      instRef = null;
    }
    if (!exiting && vm.phase !== 'error') {
      console.error(`Wrote tokenizer: ${opts.outPath}`);
    }
  }
}

/** Train **`llm tokenizer train`** with wolf-tUI progress when **`stderr`** is interactive. */
export async function runTokenizerTrain(opts: RunTokenizerTrainOpts): Promise<void> {
  const diskStream = opts.diskStream === true;

  if (!useRichTokenizerTui()) {
    await tokenizerTrainPlain({ ...opts, diskStream });
    return;
  }

  if (diskStream) {
    await tokenizerTrainDiskStreamTui(opts);
    return;
  }

  await tokenizerTrainRamTui(opts);
}
