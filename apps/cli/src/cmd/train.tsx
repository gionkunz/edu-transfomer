import { render } from '@wolf-tui/react';
import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';

import { createParameterRegistry } from '@llm-transformer-diy/autograd';
import { discoverTrainingFiles } from '@llm-transformer-diy/io';
import {
  collectGptParameters,
  createGptModel,
  trainingBatchStart,
} from '@llm-transformer-diy/model';
import { destroy, type FloatDType } from '@llm-transformer-diy/tensor';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';
import {
  AdamW,
  Trainer,
  assertCorpusTokenIdsInVocab,
  assertModelParametersFiniteForResume,
  createLmBatch,
  hydrateAdamWFromCheckpoint,
  loadGptModelFromCheckpoint,
  loadTrainingCheckpointForResume,
  registerModelParameters,
  resolveCheckpointEvery,
  runReplayStepProbe,
  saveTrainingCheckpoint,
  shouldSavePeriodicCheckpoint,
  TRAIN_DEFAULT_CHECKPOINT_EVERY,
  trainingCheckpointPaths,
  type TrainerMicroBatch,
} from '@llm-transformer-diy/training';

import { assertRuntimeSupportsComputeDtype } from '../compute-dtype';
import { resolveTrainConfig } from '../config-presets';
import { createTensorRuntime } from '../runtime';
import { primeTuiTerminalCanvas } from '../tui-prime';
import {
  buildCorpusTokenCacheFingerprint,
  corpusTokenCacheArtifactPaths,
  readCorpusTokenCache,
  writeCachedCorpusTokens,
} from './corpus-token-cache';
import { perRegionPessimisticEta } from './encode-eta';
import {
  createInitialTrainDashboardDraft,
  createTrainDashboardThrottle,
  finalizeTrainDashboardDraft,
  type TrainDashboardDraft,
} from './train-dash-model';
import { TrainDashboardView } from './train-dashboard';
import {
  TRAIN_METRICS_FILE,
  TrainMetricsLog,
  readTrainMetrics,
  type TrainMetricsRecord,
} from './train-metrics-log';

/** Defaults for **`llm train`** (override with flags). */
export const TRAIN_DEFAULT_LR_MAX = 1e-4;
export const TRAIN_DEFAULT_LR_MIN = 1e-5;
export const TRAIN_DEFAULT_WEIGHT_DECAY = 0.01;
export const TRAIN_DEFAULT_MAX_GRAD_NORM = 0.5;
/** Fraction of **`--steps`** used for linear LR warmup before cosine decay. */
export const TRAIN_DEFAULT_WARMUP_FRACTION = 0.1;
/** Sequences per optimizer step (gradient accumulation). */
export const TRAIN_DEFAULT_ACCUM = 1;
/** Validation cadence in optimizer steps when **`--val-data`** is given. */
export const TRAIN_DEFAULT_EVAL_EVERY = 250;
/** Held-out sequences averaged per validation measurement. */
export const TRAIN_DEFAULT_EVAL_BATCHES = 16;

export {
  TRAIN_DEFAULT_CHECKPOINT_EVERY,
  resolveCheckpointEvery,
} from '@llm-transformer-diy/training';

/**
 * Warmup step count from **`totalSteps × warmupFraction`**, clamped to **`[1, totalSteps - 1]`**
 * so short runs still warm up and the cosine phase always has at least one step.
 */
export function resolveWarmupSteps(totalSteps: number, warmupFraction: number): number {
  if (!Number.isInteger(totalSteps) || totalSteps <= 0) {
    throw new Error(`train: total steps must be a positive integer (got ${String(totalSteps)})`);
  }
  if (!Number.isFinite(warmupFraction) || warmupFraction <= 0 || warmupFraction >= 1) {
    throw new Error(`train: --warmup must satisfy 0 < warmup < 1 (got ${String(warmupFraction)})`);
  }
  const raw = Math.floor(totalSteps * warmupFraction);
  return Math.max(1, Math.min(raw, totalSteps - 1));
}

export type RunTrainCommandOpts = {
  readonly dataGlobs: string[];
  readonly outDir: string;
  readonly tokenizerPath: string;
  readonly configArg: string;
  readonly steps: number;
  readonly seqLen: number;
  readonly tokenCacheBase?: string;
  readonly rebuildTokenCache?: boolean;
  /** Peak LR after linear warmup; cosine decay toward **`lrMin`**. */
  readonly lrMax: number;
  readonly lrMin: number;
  /** AdamW decoupled weight decay **`λ`** (0 disables). */
  readonly weightDecay: number;
  /**
   * Global L2 grad clip threshold; **`0`** or **`Infinity`** disables clipping (risky on long runs).
   */
  readonly maxGradNorm: number;
  /** Linear warmup as a fraction of **`steps`** (default **`0.1`**). */
  readonly warmupFraction: number;
  /** Forward / parameter compute dtype (`'f32'` default). */
  readonly computeDtype: FloatDType;
  /** Save model + optimizer every N completed steps (`0` = final only). */
  readonly checkpointEvery: number;
  /** Continue from **`model.bin`** + **`train-state.json`** in **`--out`**. */
  readonly resume?: boolean;
  /** Fast-forward to this step and run forward layer probe (`--replay-only`). */
  readonly replayStep?: number;
  /** Exit after **`--replay-step`** probe instead of continuing training. */
  readonly replayOnly?: boolean;
  /** Sequence row indices to inspect during replay probe (default: 0, mid, last). */
  readonly replayRows?: readonly number[];
  /** Sequences per optimizer step; gradients are averaged over them (default **`1`**). */
  readonly accum?: number;
  /** Held-out text globs; enables the validation loss curve. */
  readonly valGlobs?: readonly string[];
  /** Measure validation loss every N optimizer steps (default **`250`**). */
  readonly evalEvery?: number;
  /** Held-out sequences averaged per measurement (default **`16`**). */
  readonly evalBatches?: number;
};

function resolvePositiveInt(value: number | undefined, fallback: number, flag: string): number {
  const v = value ?? fallback;
  if (!Number.isInteger(v) || v <= 0) {
    throw new Error(`train: ${flag} must be a positive integer (got ${String(v)})`);
  }
  return v;
}

function resolveTrainOptimHyperparams(opts: RunTrainCommandOpts): {
  lrMax: number;
  lrMin: number;
  weightDecay: number;
  maxGradNorm: number;
} {
  const { lrMax, lrMin, weightDecay, maxGradNorm } = opts;

  if (!Number.isFinite(lrMax) || !Number.isFinite(lrMin) || !Number.isFinite(weightDecay)) {
    throw new Error(
      'train: lr-max, lr-min, and weight-decay must be finite numbers (check CLI flags)',
    );
  }
  if (!Number.isFinite(maxGradNorm) && maxGradNorm !== Number.POSITIVE_INFINITY) {
    throw new Error('train: max-grad-norm must be finite or Infinity (check CLI flags)');
  }

  if (!(lrMax > 0)) {
    throw new Error(`train: lr-max must be > 0 (got ${String(lrMax)})`);
  }
  if (!(lrMin >= 0) || !(lrMin <= lrMax)) {
    throw new Error(
      `train: lr-min must satisfy 0 ≤ lr-min ≤ lr-max (got ${String(lrMin)} vs max ${String(lrMax)})`,
    );
  }
  if (!(weightDecay >= 0)) {
    throw new Error(`train: weight-decay must be ≥ 0 (got ${String(weightDecay)})`);
  }
  const clipOff = maxGradNorm === 0 || maxGradNorm === Number.POSITIVE_INFINITY;
  if (!clipOff) {
    if (!(maxGradNorm > 0)) {
      throw new Error(
        `train: max-grad-norm must be > 0, or 0/Infinity to disable clipping (got ${String(maxGradNorm)})`,
      );
    }
  }

  return {
    lrMax,
    lrMin,
    weightDecay,
    maxGradNorm: clipOff ? Number.POSITIVE_INFINITY : maxGradNorm,
  };
}

export async function runTrainCommand(opts: RunTrainCommandOpts): Promise<void> {
  const computeDtype = opts.computeDtype;
  const optim = resolveTrainOptimHyperparams(opts);
  const accum = resolvePositiveInt(opts.accum, TRAIN_DEFAULT_ACCUM, '--accum');
  const evalEvery = resolvePositiveInt(opts.evalEvery, TRAIN_DEFAULT_EVAL_EVERY, '--eval-every');
  const evalBatches = resolvePositiveInt(
    opts.evalBatches,
    TRAIN_DEFAULT_EVAL_BATCHES,
    '--eval-batches',
  );
  if (accum > 1 && opts.replayStep !== undefined) {
    throw new Error('train: --replay-step replays single-sequence steps; use it with --accum 1');
  }
  const outAbsDisplay = path.resolve(opts.outDir);
  const draft: TrainDashboardDraft = createInitialTrainDashboardDraft({
    outDirDisplay: outAbsDisplay,
    configTag: opts.configArg,
    seqLen: opts.seqLen,
    totalTrainSteps: opts.steps,
  });

  type RenderInstance = ReturnType<typeof render>;
  const dashRef: { inst?: RenderInstance } = {};
  const toggleTrainingPause = (): void => {
    if (draft.phase !== 'training') return;
    const nowMs = Date.now();
    if (draft.trainingPaused) {
      draft.trainingPaused = false;
      if (draft.trainingPauseBeganAtMs !== null) {
        draft.trainingPausedAccumMs += nowMs - draft.trainingPauseBeganAtMs;
        draft.trainingPauseBeganAtMs = null;
      }
    } else {
      draft.trainingPaused = true;
      draft.trainingPauseBeganAtMs = nowMs;
    }
    dashThrottle.flush();
  };
  const paintDashboard = (): void => {
    if (dashRef.inst === undefined) return;
    dashRef.inst.rerender(
      <TrainDashboardView
        snapshot={finalizeTrainDashboardDraft(draft)}
        onToggleTrainingPause={toggleTrainingPause}
      />,
    );
  };
  const dashThrottle = createTrainDashboardThrottle(96, paintDashboard);

  draft.phase = 'starting';
  draft.detailLine = 'mounting dashboard';
  primeTuiTerminalCanvas();
  dashRef.inst = render(
    <TrainDashboardView
      snapshot={finalizeTrainDashboardDraft(draft)}
      onToggleTrainingPause={toggleTrainingPause}
    />,
    {
      exitOnCtrlC: true,
      patchConsole: true,
    },
  );
  dashThrottle.flush();

  draft.phase = 'gpu_init';
  draft.detailLine = 'WebGPU adapter + tensor runtime';
  console.error('train: initializing WebGPU runtime…');
  dashThrottle.flush();
  const rt = await createTensorRuntime();
  assertRuntimeSupportsComputeDtype(rt.ctx, computeDtype, 'train');
  draft.gpuAdapterSummary = rt.ctx.adapterLabel;
  draft.detailLine = rt.ctx.adapterLabel;
  console.error(`train: WebGPU adapter: ${rt.ctx.adapterLabel}`);
  console.error(`train: compute dtype: ${computeDtype}`);
  if (rt.ctx.isLikelyCpuSoftwareAdapter) {
    const stderrMsg =
      'WebGPU reports a software adapter (e.g. LLVMpipe). Work runs on CPU. For NVIDIA on Linux: install the proprietary driver + Vulkan ICD so `vulkaninfo` lists your GPU (not only llvmpipe). You can point loaders at the NVIDIA ICD with e.g. VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json (path varies). On Windows use native Node; WSL often falls back to software unless GPU + Vulkan are wired for your stack.';
    draft.gpuSoftwareWarning =
      'Software WebGPU (LLVMpipe/SwiftShader): training is CPU-bound. Fix OS Vulkan/ICD or run native Windows — details on stderr.';
    console.error(`train: ${stderrMsg}`);
  } else {
    draft.gpuSoftwareWarning = null;
  }
  dashThrottle.flush();

  draft.phase = 'tokenizer_load';
  draft.detailLine = opts.tokenizerPath;
  dashThrottle.flush();
  const tokenizer = new BPETokenizer();
  await tokenizer.load(opts.tokenizerPath);
  draft.detailLine = `${opts.tokenizerPath} · vocab ${String(tokenizer.vocabSize)}`;
  dashThrottle.flush();

  draft.phase = 'resolve_config';
  const vocabSize = tokenizer.vocabSize;
  draft.detailLine = `preset ${opts.configArg} · V=${String(vocabSize)}`;
  dashThrottle.flush();
  const gptCfg = resolveTrainConfig(opts.configArg, vocabSize);

  draft.phase = 'discover_files';
  draft.detailLine = `--data globs → paths`;
  draft.fileProgress = null;
  dashThrottle.flush();
  const files = await discoverTrainingFiles(opts.dataGlobs);
  if (files.length === 0) {
    throw new Error('train: --data matched no files');
  }
  draft.detailLine = `${String(files.length)} file(s)`;
  dashThrottle.flush();

  const cacheBaseTrimmed = opts.tokenCacheBase?.trim();
  const corpusTokenFingerprint =
    cacheBaseTrimmed !== undefined && cacheBaseTrimmed.length > 0
      ? await buildCorpusTokenCacheFingerprint({
          tokenizerPath: opts.tokenizerPath,
          corpusRelPaths: files,
        })
      : null;

  let tokenIds: Uint32Array | undefined;
  let corpusUtf16Units = 0;

  if (
    corpusTokenFingerprint !== null &&
    cacheBaseTrimmed !== undefined &&
    cacheBaseTrimmed.length > 0 &&
    !opts.rebuildTokenCache
  ) {
    const cacheResult = await readCorpusTokenCache(cacheBaseTrimmed, corpusTokenFingerprint);
    if (cacheResult.status === 'hit') {
      tokenIds = cacheResult.ids;
      corpusUtf16Units = cacheResult.corpusUtf16Units;

      draft.phase = 'read_corpus';
      draft.detailLine = 'skipped (token cache)';
      draft.fileProgress = {
        currentIndex: files.length,
        totalFiles: files.length,
        pathRel:
          files.length === 1 && files[0] !== undefined ? files[0] : `${String(files.length)} files`,
        cumulativeChars: corpusUtf16Units,
      };
      dashThrottle.flush();

      draft.phase = 'encode_corpus';
      draft.detailLine = `${String(corpusUtf16Units)} chars → skipped (disk cache)`;
      draft.encodeProgress = null;
      dashThrottle.flush();

      const cachePaths = corpusTokenCacheArtifactPaths(cacheBaseTrimmed);
      console.error(
        `train: token cache hit → ${String(tokenIds.length)} ids · ${String(files.length)} corpus file(s) · ${cachePaths.idsBin}`,
      );
    } else {
      console.error(`train: token cache miss (${cacheBaseTrimmed}):`);
      for (const reason of cacheResult.reasons) {
        console.error(`  ${reason}`);
      }
      console.error('train: re-encoding corpus');
    }
  } else if (
    opts.rebuildTokenCache &&
    cacheBaseTrimmed !== undefined &&
    cacheBaseTrimmed.length > 0
  ) {
    console.error(
      `train: token cache bypass (--rebuild-token-cache) → will overwrite ${cacheBaseTrimmed}.{meta.json,ids}`,
    );
  }

  if (!tokenIds) {
    draft.phase = 'read_corpus';
    draft.detailLine =
      files.length === 1
        ? (files[0] ?? '')
        : `sequential per-file encode · ${String(files.length)} files`;
    draft.fileProgress = {
      currentIndex: 0,
      totalFiles: files.length,
      pathRel: files.length === 1 && files[0] !== undefined ? files[0] : '(per-file reads)',
      cumulativeChars: 0,
    };
    dashThrottle.flush();

    // Per-file read+encode loop. Two reasons we cannot read+join all files first:
    //   1. V8's max single-string length is ~512 MiB; large multi-shard corpora
    //      (e.g. multiple Cosmopedia shards) blow this on `textParts.join('\n')`.
    //   2. Holding every file's UTF-16 string + the joined string + the encoder
    //      working set simultaneously needlessly inflates peak RSS.
    // Equivalence: under the `gpt2-v1` pretokenizer (and any pretokenizer that
    // splits on `\n`), `encode(f0 + '\n' + f1) === concat(encode(f0),
    // encode('\n'), encode(f1))`. The tokenizer test
    // `encodeUint32(x) === concat(encodeUint32(region_i))` asserts this. We
    // therefore encode each file independently and splice the `\n`-token stream
    // between adjacent files, preserving exact token-id equivalence with the
    // old joined-corpus path so existing on-disk caches remain valid.
    draft.phase = 'encode_corpus';
    draft.detailLine = `${String(files.length)} file(s) → token ids`;
    draft.encodeProgress = null;
    dashThrottle.flush();

    const newlineSeparatorIds =
      files.length > 1 ? tokenizer.encodeUint32('\n') : new Uint32Array(0);

    const encodeWallStartMs = Date.now();
    const pretokenizerPolicy = tokenizer.pretokenizer;

    // Cross-file accumulators for stable EMA / ETA over the whole corpus.
    let lastSampleWallMs = encodeWallStartMs;
    let passesPerSecEma: number | null = null;
    let cumulativePassesAcrossRegions = 0;
    let completedRegionsGlobal = 0;
    let expandCharsCurr = 0;
    let expandBytesCurr = 0;
    let initialTokForFile = 0;

    const tokenChunks: Uint32Array[] = [];
    let cumulativeChars = 0;
    let totalTokenCountSoFar = 0;

    for (let fi = 0; fi < files.length; fi++) {
      const rel = files[fi];
      if (rel === undefined) continue;

      draft.fileProgress = {
        currentIndex: fi,
        totalFiles: files.length,
        pathRel: rel,
        cumulativeChars,
      };
      dashThrottle.flush();

      const text = await readFile(path.resolve(rel), 'utf8');
      cumulativeChars += text.length;
      draft.fileProgress = {
        currentIndex: fi + 1,
        totalFiles: files.length,
        pathRel: rel,
        cumulativeChars,
      };
      dashThrottle.flush();

      console.error(
        `train: encoding file ${String(fi + 1)}/${String(files.length)} (${String(text.length)} chars): ${rel}`,
      );

      // Per-file region tracking; rolled into the global accumulators at end of file.
      let lastRegionIndexLocal = -1;
      let lastPassCountForRegionLocal = 0;

      const fileIds = await tokenizer.encodeUint32Async(text, {
        onYield: () => {
          dashThrottle.schedule();
        },
        onProgress: (ev) => {
          const nowMs = Date.now();
          const encodeElapsedSec = (nowMs - encodeWallStartMs) / 1000;
          if (ev.phase === 'expand') {
            expandCharsCurr = ev.textChars;
            expandBytesCurr = ev.byteLength;
            initialTokForFile = ev.initialTokenCount;
            lastSampleWallMs = nowMs;
            lastRegionIndexLocal = -1;
            lastPassCountForRegionLocal = 0;
            const meanPasses =
              completedRegionsGlobal > 0
                ? cumulativePassesAcrossRegions / completedRegionsGlobal
                : null;
            draft.encodeProgress = {
              textChars: ev.textChars,
              byteLength: ev.byteLength,
              mergeIteration: 0,
              mergeSubphase: 'expand',
              tokenCount: totalTokenCountSoFar + ev.initialTokenCount,
              initialTokenCount: initialTokForFile,
              encodeElapsedSec,
              pretokenizerPolicy,
              regionsPerSec: null,
              etaSec: null,
              regionIndex: 0,
              regionsTotal: 0,
              meanPassesPerRegion: meanPasses,
            };
          } else {
            if (ev.regionIndex !== lastRegionIndexLocal) {
              if (lastRegionIndexLocal >= 0) {
                cumulativePassesAcrossRegions += lastPassCountForRegionLocal;
                completedRegionsGlobal += 1;
              }
              lastRegionIndexLocal = ev.regionIndex;
              lastPassCountForRegionLocal = 0;
            }
            lastPassCountForRegionLocal = ev.mergeIteration;

            const dWall = (nowMs - lastSampleWallMs) / 1000;
            if (dWall >= 0.05) {
              const instant = 1 / dWall;
              passesPerSecEma =
                passesPerSecEma === null ? instant : instant * 0.25 + passesPerSecEma * 0.75;
              lastSampleWallMs = nowMs;
            }

            const meanPasses =
              completedRegionsGlobal > 0
                ? cumulativePassesAcrossRegions / completedRegionsGlobal
                : null;
            const regionsRemaining = Math.max(0, ev.regionsTotal - ev.regionIndex);
            const eta = perRegionPessimisticEta(regionsRemaining, meanPasses, passesPerSecEma);
            const regionsPerSec =
              meanPasses !== null && passesPerSecEma !== null && meanPasses > 0
                ? passesPerSecEma / meanPasses
                : null;

            draft.encodeProgress = {
              textChars: expandCharsCurr,
              byteLength: expandBytesCurr,
              mergeIteration: ev.mergeIteration,
              mergeSubphase: 'merge',
              tokenCount: totalTokenCountSoFar + ev.tokenCount,
              initialTokenCount: initialTokForFile,
              encodeElapsedSec,
              pretokenizerPolicy,
              regionsPerSec,
              etaSec: eta,
              regionIndex: ev.regionIndex,
              regionsTotal: ev.regionsTotal,
              meanPassesPerRegion: meanPasses,
            };
          }
          dashThrottle.schedule();
        },
      });

      if (lastRegionIndexLocal >= 0) {
        cumulativePassesAcrossRegions += lastPassCountForRegionLocal;
        completedRegionsGlobal += 1;
      }

      tokenChunks.push(fileIds);
      totalTokenCountSoFar += fileIds.length;
      if (fi < files.length - 1) {
        tokenChunks.push(newlineSeparatorIds);
        totalTokenCountSoFar += newlineSeparatorIds.length;
        cumulativeChars += 1;
      }
    }
    dashThrottle.flush();

    let totalTokenCount = 0;
    for (const c of tokenChunks) totalTokenCount += c.length;
    tokenIds = new Uint32Array(totalTokenCount);
    {
      let off = 0;
      for (const c of tokenChunks) {
        tokenIds.set(c, off);
        off += c.length;
      }
    }

    draft.encodeProgress = null;
    corpusUtf16Units = cumulativeChars;

    if (
      corpusTokenFingerprint !== null &&
      cacheBaseTrimmed !== undefined &&
      cacheBaseTrimmed.length > 0
    ) {
      await writeCachedCorpusTokens(
        cacheBaseTrimmed,
        corpusTokenFingerprint,
        tokenIds,
        corpusUtf16Units,
      );
      console.error(
        `train: wrote token cache (${String(tokenIds.length)} ids): ${cacheBaseTrimmed}.{meta.json,ids}`,
      );
    }
  }

  if (!tokenIds) {
    throw new Error('train: internal error — token ids were not resolved');
  }

  draft.corpusChars = corpusUtf16Units;

  draft.phase = 'assemble_corpus';
  draft.detailLine = `${String(tokenIds.length)} tokens`;
  draft.corpusTokens = tokenIds.length;
  dashThrottle.flush();

  console.error(
    `train: ${String(tokenIds.length)} tokens · seqLen=${String(opts.seqLen)} · steps=${String(opts.steps)}`,
  );

  draft.phase = 'verify_corpus';
  draft.detailLine =
    tokenIds.length > opts.seqLen ? 'bounds OK for seq slicing' : 'too few tokens vs seqLen';
  dashThrottle.flush();
  if (tokenIds.length <= opts.seqLen) {
    throw new Error(
      `train: need more tokens than seqLen (${String(tokenIds.length)} <= ${String(opts.seqLen)})`,
    );
  }

  assertCorpusTokenIdsInVocab(tokenIds, gptCfg.vocabSize);

  // Held-out text: never trained on, so its loss shows generalization rather
  // than memorization. Encoded in memory (it is small) and not cached.
  let valTokenIds: Uint32Array | null = null;
  if (opts.valGlobs !== undefined && opts.valGlobs.length > 0) {
    const valFiles = await discoverTrainingFiles([...opts.valGlobs]);
    if (valFiles.length === 0) {
      throw new Error('train: --val-data matched no files');
    }
    const trainSet = new Set(files.map((f) => path.resolve(f)));
    const overlap = valFiles.find((f) => trainSet.has(path.resolve(f)));
    if (overlap !== undefined) {
      throw new Error(`train: --val-data file is also in --data (not held out): ${overlap}`);
    }
    const chunks: Uint32Array[] = [];
    let total = 0;
    for (let fi = 0; fi < valFiles.length; fi++) {
      const rel = valFiles[fi];
      if (rel === undefined) continue;
      draft.detailLine = `encoding validation file ${String(fi + 1)}/${String(valFiles.length)}: ${rel}`;
      dashThrottle.flush();
      const text = await readFile(path.resolve(rel), 'utf8');
      const ids = await tokenizer.encodeUint32Async(text, {
        onYield: () => {
          dashThrottle.schedule();
        },
      });
      chunks.push(ids);
      total += ids.length;
    }
    valTokenIds = new Uint32Array(total);
    let off = 0;
    for (const c of chunks) {
      valTokenIds.set(c, off);
      off += c.length;
    }
    if (valTokenIds.length <= opts.seqLen + 1) {
      throw new Error(
        `train: validation data has ${String(valTokenIds.length)} tokens, need more than seqLen + 1`,
      );
    }
    assertCorpusTokenIdsInVocab(valTokenIds, gptCfg.vocabSize);
    console.error(
      `train: validation ${String(valTokenIds.length)} tokens · every ${String(evalEvery)} steps · ${String(evalBatches)} sequences`,
    );
  }

  const totalSteps = opts.steps;
  const warmupSteps = resolveWarmupSteps(totalSteps, opts.warmupFraction);
  const schedule = {
    lrMax: optim.lrMax,
    lrMin: optim.lrMin,
    warmupSteps,
    totalSteps,
  };
  const checkpointEvery = resolveCheckpointEvery(totalSteps, opts.checkpointEvery);

  let startStep = 0;
  let model;

  if (opts.resume) {
    draft.phase = 'build_model';
    draft.detailLine = 'loading checkpoint for --resume';
    dashThrottle.flush();
    const resumeInfo = await loadTrainingCheckpointForResume({
      outDir: outAbsDisplay,
      config: gptCfg,
      totalSteps,
      seqLen: opts.seqLen,
      computeDtype,
    });
    startStep = resumeInfo.completedStep;
    const ckptPaths = trainingCheckpointPaths(outAbsDisplay);
    const loaded = await loadGptModelFromCheckpoint(rt, ckptPaths.modelBin, ckptPaths.modelJson, {
      computeDtype,
    });
    model = loaded.model;
    await assertModelParametersFiniteForResume(rt, model, `step ${String(startStep)}`);
    console.error(
      `train: resuming from step ${String(startStep)}/${String(totalSteps)} (${ckptPaths.modelJson})`,
    );
  } else {
    draft.phase = 'build_model';
    draft.detailLine = `layers=${String(gptCfg.nLayers)} d_model inferred`;
    dashThrottle.flush();
    model = createGptModel(rt, gptCfg, { dtype: computeDtype });
  }

  const reg = createParameterRegistry();
  registerModelParameters(reg, collectGptParameters(model));
  const opt = new AdamW(rt, reg, { lr: optim.lrMax, weightDecay: optim.weightDecay });

  if (opts.resume) {
    const ckptPaths = trainingCheckpointPaths(outAbsDisplay);
    await hydrateAdamWFromCheckpoint(
      rt,
      opt,
      model,
      ckptPaths.optimizerBin,
      ckptPaths.optimizerJson,
    );
  }

  draft.phase = 'trainer_init';
  const ckptLabel =
    checkpointEvery > 0 ? `ckpt every ${String(checkpointEvery)}` : 'ckpt final only';
  draft.detailLine = `AdamW · lr ${String(optim.lrMax)}→${String(optim.lrMin)} · warmup ${String(warmupSteps)}/${String(totalSteps)} (${String(opts.warmupFraction)}) · wd ${String(optim.weightDecay)} · clip ${Number.isFinite(optim.maxGradNorm) ? String(optim.maxGradNorm) : 'off'} · ${ckptLabel}`;
  dashThrottle.flush();
  const trainer = new Trainer(rt, model, reg, opt, {
    maxGradNorm: optim.maxGradNorm,
    schedule,
    // Smooth out V8's lazy GC of stale GPUBuffer wrappers when Node was
    // started with `--expose-gc`. No-op otherwise. The buffer pool already
    // eliminates most per-step churn; this is belt-and-suspenders for long runs.
    forceGcEverySteps: 50,
    // Always read the gradient norm so the dashboard / metrics.jsonl have it
    // even when clipping is off (one small readback per step).
    measureGradNorm: true,
    // `logBufferPoolEverySteps` is intentionally not set: the pool's hit-rate
    // and idle-footprint behaviour is well understood and the per-100-step
    // stderr line just clutters the diagnostics stream. Re-enable on demand
    // when investigating pool regressions.
  });

  draft.lrSchedule = schedule;
  draft.accum = accum;
  draft.maxGradNorm = Number.isFinite(optim.maxGradNorm) ? optim.maxGradNorm : null;
  draft.uniformLoss = Math.log(gptCfg.vocabSize);
  draft.archSummary = `${String(gptCfg.nLayers)}L · d${String(model.config.dModel)} · V${String(model.config.vocabSize)} · ${computeDtype}`;
  draft.phase = 'training';
  draft.trainingPhaseStartedAtMs = Date.now();
  draft.trainLrDisplay = schedule.lrMax;
  draft.detailLine = 'optimizer stepping (GPU shader warmup may pause step 1)';
  dashThrottle.flush();

  console.error(
    `train: optim lr_max=${String(optim.lrMax)} lr_min=${String(optim.lrMin)} warmup=${String(warmupSteps)}/${String(totalSteps)} (fraction=${String(opts.warmupFraction)}) weight_decay=${String(optim.weightDecay)} max_grad_norm=${Number.isFinite(optim.maxGradNorm) ? String(optim.maxGradNorm) : 'off'} accum=${String(accum)} checkpoint_every=${String(checkpointEvery)}`,
  );
  console.error(
    'train: first optimizer step often waits on GPU shader/pipeline warmup (minutes on some setups).',
  );

  draft.fileProgress = null;

  if (opts.replayStep !== undefined) {
    if (!Number.isInteger(opts.replayStep) || opts.replayStep < 0) {
      throw new Error(`train: --replay-step must be a non-negative integer`);
    }
    if (opts.replayStep >= totalSteps) {
      throw new Error(
        `train: --replay-step ${String(opts.replayStep)} must be < --steps ${String(totalSteps)}`,
      );
    }
    if (opts.replayStep < startStep) {
      throw new Error(
        `train: --replay-step ${String(opts.replayStep)} is before resume point ${String(startStep)}`,
      );
    }
    if (opts.replayOnly) {
      console.error(
        `train: replay-only mode → run steps ${String(startStep + 1)}..${String(opts.replayStep)} then forward probe`,
      );
      draft.phase = 'training';
      dashThrottle.flush();
      await runReplayStepProbe({
        rt,
        model,
        trainer,
        tokenIds,
        seqLen: opts.seqLen,
        vocabSize: gptCfg.vocabSize,
        fromStep: startStep,
        replayStep: opts.replayStep,
        ...(opts.replayRows !== undefined && opts.replayRows.length > 0
          ? { rowsToScan: opts.replayRows }
          : {}),
      });
      draft.phase = 'done';
      draft.detailLine = `replay probe step ${String(opts.replayStep)}`;
      dashThrottle.flush();
      dashRef.inst?.unmount();
      opt.dispose();
      return;
    }
  }

  const writeTrainingCheckpoint = async (
    completedStep: number,
    trainLossLast: number,
  ): Promise<void> => {
    draft.phase = 'save_checkpoint';
    draft.detailLine = `step ${String(completedStep)}/${String(totalSteps)}`;
    draft.saveProgress = null;
    dashThrottle.flush();
    await mkdir(outAbsDisplay, { recursive: true });
    await saveTrainingCheckpoint({
      rt,
      model,
      optimizer: opt,
      outDir: outAbsDisplay,
      config: gptCfg,
      completedStep,
      totalSteps,
      seqLen: opts.seqLen,
      computeDtype,
      schedule,
      weightDecay: optim.weightDecay,
      maxGradNorm: optim.maxGradNorm,
      trainLossLast,
      tokenizerSourcePath: opts.tokenizerPath,
      onCheckpointTensor: (info) => {
        draft.saveProgress = {
          tensorIndex: info.index,
          totalTensors: info.total,
          tensorName: info.name,
        };
        dashThrottle.flush();
      },
    });
    draft.saveProgress = null;
    draft.phase = 'training';
    draft.detailLine = 'optimizer stepping';
    dashThrottle.flush();
    console.error(`train: checkpoint saved at step ${String(completedStep)}/${String(totalSteps)}`);
  };

  if (startStep > 0) {
    draft.trainStepCompleted = startStep;
  }

  // metrics.jsonl: one line per optimizer step, next to the checkpoint. On
  // --resume the history up to the checkpointed step is kept and re-plotted.
  await mkdir(outAbsDisplay, { recursive: true });
  const metricsPath = path.join(outAbsDisplay, TRAIN_METRICS_FILE);
  const keptMetrics: TrainMetricsRecord[] = opts.resume
    ? (await readTrainMetrics(metricsPath)).filter((r) => r.step <= startStep)
    : [];
  for (const r of keptMetrics) {
    draft.lossHistory.push(r.step, r.loss);
    if (r.gradNorm !== null) draft.gradNormHistory.push(r.step, r.gradNorm);
    if (r.valLoss !== undefined) draft.valHistory.push({ step: r.step, loss: r.valLoss });
  }
  const metricsLog = await TrainMetricsLog.open(metricsPath, keptMetrics);

  /**
   * Mean loss over **`evalBatches`** held-out sequences. The windows are fixed
   * (evenly spaced through the validation tokens), so every measurement scores
   * the same text and the curve is comparable across steps and across runs.
   */
  const measureValLoss = async (): Promise<number | null> => {
    if (valTokenIds === null) return null;
    const maxStart = valTokenIds.length - opts.seqLen - 1;
    const n = Math.min(evalBatches, maxStart + 1);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const start = n === 1 ? 0 : Math.floor((i * maxStart) / (n - 1));
      const { input, targets } = createLmBatch(rt, valTokenIds, start, opts.seqLen);
      try {
        sum += await trainer.evalLoss(input, targets);
      } finally {
        destroy(input);
        destroy(targets);
      }
    }
    return sum / n;
  };

  try {
    for (let s = startStep; s < totalSteps; s++) {
      while (draft.trainingPaused) {
        await new Promise<void>((r) => {
          globalThis.setTimeout(r, 120);
        });
        dashThrottle.schedule();
      }
      if (s === startStep) {
        console.error(`train: step ${String(s + 1)}/${String(totalSteps)}…`);
      }
      // `accum` sequences per optimizer step. With accum = 1 the offsets are
      // exactly `trainingBatchStart(…, s)`, which `--replay-step` relies on.
      const microBatches: TrainerMicroBatch[] = [];
      for (let m = 0; m < accum; m++) {
        const start = trainingBatchStart(tokenIds.length, opts.seqLen, s * accum + m);
        const { input, targets } = createLmBatch(rt, tokenIds, start, opts.seqLen);
        microBatches.push({
          input,
          targets,
          batch: { batchStart: start, vocabSize: gptCfg.vocabSize },
        });
      }
      const t0 = performance.now();
      let stepResult;
      try {
        stepResult = await trainer.trainStepAccumulated(microBatches, s);
      } finally {
        for (const mb of microBatches) {
          destroy(mb.input);
          destroy(mb.targets);
        }
      }
      const loss = stepResult.loss;
      const dt = Math.max(1e-6, (performance.now() - t0) / 1000);
      const tokPerSec = (opts.seqLen * accum) / dt;

      const stepDone = s + 1;
      const valLoss =
        stepDone % evalEvery === 0 || stepDone === totalSteps ? await measureValLoss() : null;

      draft.trainStepCompleted = stepDone;
      draft.trainLossLast = loss;
      draft.trainLrDisplay = stepResult.lr;
      draft.tokensPerSecApprox = tokPerSec;
      draft.recentLosses.push(loss);
      draft.lossHistory.push(stepDone, loss);
      draft.gradNormLast = stepResult.gradNorm;
      if (stepResult.gradNorm !== null) {
        draft.gradNormHistory.push(stepDone, stepResult.gradNorm);
      }
      if (valLoss !== null) {
        draft.valHistory.push({ step: stepDone, loss: valLoss });
      }
      metricsLog.write({
        step: stepDone,
        loss,
        lr: stepResult.lr,
        gradNorm: stepResult.gradNorm,
        tokPerSec,
        ...(valLoss !== null ? { valLoss } : {}),
      });
      dashThrottle.schedule();

      const completedStep = s + 1;
      if (shouldSavePeriodicCheckpoint(completedStep, totalSteps, checkpointEvery)) {
        await writeTrainingCheckpoint(completedStep, loss);
      }

      // Drain Node's global perf-entry buffer every ~256 steps.
      // bindings (dawn-node) emit a `performance.measure(...)` per kernel
      // dispatch; over very long runs (e.g. 20k steps × ~50 dispatches) this
      // buffer overflows the default 10k cap and Node prints
      // `MaxPerformanceEntryBufferExceededWarning`. We never read marks /
      // measures from our own code (only `performance.now()`), so it's safe
      // to drop them. Cheap: O(buffer size) once per N steps.
      if ((s & 0xff) === 0xff) {
        performance.clearMarks();
        performance.clearMeasures();
      }
    }
    draft.trainingPaused = false;
    draft.trainingPauseBeganAtMs = null;
    draft.trainingWallStoppedAtMs = Date.now();
    dashThrottle.flush();

    draft.phase = 'save_checkpoint';
    draft.detailLine = 'model.bin + model.json + optimizer';
    draft.saveProgress = null;
    dashThrottle.flush();

    const finalLoss = draft.trainLossLast;
    if (finalLoss === null || !Number.isFinite(finalLoss)) {
      throw new Error('train: internal error — no finite final loss before checkpoint save');
    }
    await writeTrainingCheckpoint(totalSteps, finalLoss);

    draft.phase = 'finalize';
    draft.detailLine = 'done';

    draft.phase = 'done';
    draft.detailLine = 'artifacts written';
    dashThrottle.flush();
  } finally {
    await metricsLog.close();
    dashRef.inst?.unmount();
    opt.dispose();
  }
}
