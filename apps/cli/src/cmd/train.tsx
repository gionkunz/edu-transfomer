import { render } from '@wolf-tui/react';
import { copyFile, mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';

import { createParameterRegistry } from '@llm-transformer-diy/autograd';
import { discoverTrainingFiles } from '@llm-transformer-diy/io';
import { collectGptParameters, createGptModel } from '@llm-transformer-diy/model';
import { destroy, type FloatDType } from '@llm-transformer-diy/tensor';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';
import {
  AdamW,
  Trainer,
  assertCorpusTokenIdsInVocab,
  createLmBatch,
  getCosineLr,
  registerModelParameters,
  saveGptCheckpoint,
} from '@llm-transformer-diy/training';

import { assertRuntimeSupportsComputeDtype } from '../compute-dtype';
import { resolveTrainConfig } from '../config-presets';
import { createTensorRuntime } from '../runtime';
import { primeTuiTerminalCanvas } from '../tui-prime';
import {
  buildCorpusTokenCacheFingerprint,
  tryReadCachedCorpusTokens,
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

/** Defaults for **`llm train`** (override with flags). */
export const TRAIN_DEFAULT_LR_MAX = 1e-4;
export const TRAIN_DEFAULT_LR_MIN = 1e-5;
export const TRAIN_DEFAULT_WEIGHT_DECAY = 0.01;
export const TRAIN_DEFAULT_MAX_GRAD_NORM = 0.5;

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
  /** Forward / parameter compute dtype (`'f32'` default). */
  readonly computeDtype: FloatDType;
};

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
    const hit = await tryReadCachedCorpusTokens(cacheBaseTrimmed, corpusTokenFingerprint);
    if (hit !== null) {
      tokenIds = hit.ids;
      corpusUtf16Units = hit.corpusUtf16Units;

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

      console.error(
        `train: token cache hit → ${String(tokenIds.length)} ids (${cacheBaseTrimmed}.ids)`,
      );
    }
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

  draft.phase = 'build_model';
  draft.detailLine = `layers=${String(gptCfg.nLayers)} d_model inferred`;
  dashThrottle.flush();
  const model = createGptModel(rt, gptCfg, { dtype: computeDtype });
  const reg = createParameterRegistry();
  registerModelParameters(reg, collectGptParameters(model));
  const opt = new AdamW(rt, reg, { lr: optim.lrMax, weightDecay: optim.weightDecay });

  const totalSteps = opts.steps;
  // Warmup = 10% of total steps, GPT-2/-3 style. Clamp to [1, totalSteps-1] so
  // very short runs (smoke tests) still get at least one warmup step and never
  // consume the entire schedule. The previous `min(10, …)` cap meant 1000-step
  // runs only warmed up for 10 steps, leaving the cosine to immediately decay
  // from peak LR and effectively under-training at the configured `lrMax`.
  const warmupSteps = Math.max(1, Math.min(Math.floor(totalSteps / 10), totalSteps - 1));
  const schedule = {
    lrMax: optim.lrMax,
    lrMin: optim.lrMin,
    warmupSteps,
    totalSteps,
  };

  draft.phase = 'trainer_init';
  draft.detailLine = `AdamW · lr ${String(optim.lrMax)}→${String(optim.lrMin)} · warmup ${String(warmupSteps)}/${String(totalSteps)} · wd ${String(optim.weightDecay)} · clip ${Number.isFinite(optim.maxGradNorm) ? String(optim.maxGradNorm) : 'off'}`;
  dashThrottle.flush();
  const trainer = new Trainer(rt, model, reg, opt, {
    maxGradNorm: optim.maxGradNorm,
    schedule,
    // Smooth out V8's lazy GC of stale GPUBuffer wrappers when Node was
    // started with `--expose-gc`. No-op otherwise. The buffer pool already
    // eliminates most per-step churn; this is belt-and-suspenders for long runs.
    forceGcEverySteps: 50,
    // `logBufferPoolEverySteps` is intentionally not set: the pool's hit-rate
    // and idle-footprint behaviour is well understood and the per-100-step
    // stderr line just clutters the diagnostics stream. Re-enable on demand
    // when investigating pool regressions.
  });

  draft.archSummary = `${String(gptCfg.nLayers)}L · d${String(model.config.dModel)} · V${String(model.config.vocabSize)} · ${computeDtype}`;
  draft.phase = 'training';
  draft.trainingPhaseStartedAtMs = Date.now();
  draft.trainLrDisplay = schedule.lrMax;
  draft.detailLine = 'optimizer stepping (GPU shader warmup may pause step 1)';
  dashThrottle.flush();

  console.error(
    `train: optim lr_max=${String(optim.lrMax)} lr_min=${String(optim.lrMin)} warmup=${String(warmupSteps)}/${String(totalSteps)} weight_decay=${String(optim.weightDecay)} max_grad_norm=${Number.isFinite(optim.maxGradNorm) ? String(optim.maxGradNorm) : 'off'}`,
  );
  console.error(
    'train: first optimizer step often waits on GPU shader/pipeline warmup (minutes on some setups).',
  );

  draft.fileProgress = null;

  try {
    for (let s = 0; s < totalSteps; s++) {
      while (draft.trainingPaused) {
        await new Promise<void>((r) => {
          globalThis.setTimeout(r, 120);
        });
        dashThrottle.schedule();
      }
      if (s === 0) {
        console.error('train: step 1/' + String(totalSteps) + '…');
      }
      const maxStart = tokenIds.length - opts.seqLen - 1;
      const start = Math.floor((s * 9973) % (maxStart + 1));
      const { input, targets } = createLmBatch(rt, tokenIds, start, opts.seqLen);
      const t0 = performance.now();
      const loss = await trainer.trainStep(input, targets, s);
      const dt = Math.max(1e-6, (performance.now() - t0) / 1000);
      destroy(input);
      destroy(targets);

      draft.trainStepCompleted = s + 1;
      draft.trainLossLast = loss;
      draft.trainLrDisplay = getCosineLr(s, schedule);
      draft.tokensPerSecApprox = opts.seqLen / dt;
      draft.recentLosses.push(loss);
      dashThrottle.flush();

      // Drain Node's global perf-entry buffer every ~256 steps. The WebGPU
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
    draft.detailLine = 'model.bin + model.json';
    draft.saveProgress = null;
    dashThrottle.flush();

    await mkdir(outAbsDisplay, { recursive: true });
    const binPath = path.join(outAbsDisplay, 'model.bin');
    const jsonPath = path.join(outAbsDisplay, 'model.json');
    await saveGptCheckpoint({
      rt,
      model,
      binPath,
      jsonPath,
      config: gptCfg,
      trainingStep: totalSteps,
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
    dashThrottle.flush();

    draft.phase = 'finalize';
    draft.detailLine = 'tokenizer.json';
    dashThrottle.flush();
    await copyFile(opts.tokenizerPath, path.join(outAbsDisplay, 'tokenizer.json'));

    draft.phase = 'done';
    draft.detailLine = 'artifacts written';
    dashThrottle.flush();
  } finally {
    dashRef.inst?.unmount();
    opt.dispose();
  }
}
