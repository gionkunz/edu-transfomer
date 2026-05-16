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
import {
  pessimisticEtaFromRate,
  pessimisticMergeRemaining,
  pessimisticMergeUpperBound,
  mergeProgressPct,
} from './encode-eta';
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
  if (computeDtype === 'fp16') {
    console.error(
      'train: fp16 training is unstable without loss scaling — prefer bf16 when the adapter supports it.',
    );
  }
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
      files.length === 1 ? (files[0] ?? '') : `parallel disk read · ${String(files.length)} files`;
    draft.fileProgress = {
      currentIndex: 0,
      totalFiles: files.length,
      pathRel: files.length === 1 && files[0] !== undefined ? files[0] : '(parallel reads)',
      cumulativeChars: 0,
    };
    dashThrottle.flush();

    // Whole corpus becomes one `\n`-joined UTF-16 string → BPE cannot be split across workers without changing merges.
    const textParts = await Promise.all(files.map((rel) => readFile(path.resolve(rel), 'utf8')));
    const cumulativeChars = textParts.reduce((sum, chunk) => sum + chunk.length, 0);

    draft.fileProgress = {
      currentIndex: files.length,
      totalFiles: files.length,
      pathRel:
        files.length === 1 && files[0] !== undefined ? files[0] : `${String(files.length)} files`,
      cumulativeChars,
    };
    dashThrottle.flush();

    draft.phase = 'encode_corpus';
    draft.detailLine = `${String(cumulativeChars)} chars → token ids`;
    draft.encodeProgress = null;
    dashThrottle.flush();

    const corpus = textParts.join('\n');

    console.error(
      `train: encoding corpus (${String(corpus.length)} chars); BPE merges stream progress in the dashboard…`,
    );

    let expandChars = corpus.length;
    let expandBytes = corpus.length;

    const encodeWallStartMs = Date.now();
    let initialTokForEncode = 0;
    let lastMergeSampleWallMs = encodeWallStartMs;
    let lastMergeIterForRate = 0;
    let mergesPerSecEma: number | null = null;
    let mergeRateSamples = 0;

    tokenIds = await tokenizer.encodeUint32Async(corpus, {
      onYield: () => {
        dashThrottle.schedule();
      },
      onProgress: (ev) => {
        const nowMs = Date.now();
        const encodeElapsedSec = (nowMs - encodeWallStartMs) / 1000;
        if (ev.phase === 'expand') {
          expandChars = ev.textChars;
          expandBytes = ev.byteLength;
          initialTokForEncode = ev.initialTokenCount;
          lastMergeSampleWallMs = nowMs;
          lastMergeIterForRate = 0;
          mergesPerSecEma = null;
          mergeRateSamples = 0;
          const upper = pessimisticMergeUpperBound(initialTokForEncode);
          draft.encodeProgress = {
            textChars: ev.textChars,
            byteLength: ev.byteLength,
            mergeIteration: 0,
            mergeSubphase: 'expand',
            tokenCount: ev.initialTokenCount,
            initialTokenCount: initialTokForEncode,
            encodeElapsedSec,
            pessimisticMergeUpperBound: upper,
            pessimisticMergeRemaining: pessimisticMergeRemaining(initialTokForEncode, 0),
            mergesPerSec: null,
            pessimisticEtaSec: null,
            mergeProgressPct: 0,
          };
        } else {
          const dIter = ev.mergeIteration - lastMergeIterForRate;
          const dWall = (nowMs - lastMergeSampleWallMs) / 1000;
          if (dWall >= 0.05 && dIter > 0) {
            mergeRateSamples += 1;
            const instant = dIter / dWall;
            mergesPerSecEma =
              mergesPerSecEma === null ? instant : instant * 0.25 + mergesPerSecEma * 0.75;
            lastMergeSampleWallMs = nowMs;
            lastMergeIterForRate = ev.mergeIteration;
          }
          const remain = pessimisticMergeRemaining(initialTokForEncode, ev.mergeIteration);
          const eta = pessimisticEtaFromRate(remain, mergesPerSecEma, mergeRateSamples, 2);
          draft.encodeProgress = {
            textChars: expandChars,
            byteLength: expandBytes,
            mergeIteration: ev.mergeIteration,
            mergeSubphase: 'merge',
            tokenCount: ev.tokenCount,
            initialTokenCount: initialTokForEncode,
            encodeElapsedSec,
            pessimisticMergeUpperBound: pessimisticMergeUpperBound(initialTokForEncode),
            pessimisticMergeRemaining: remain,
            mergesPerSec: mergesPerSecEma,
            pessimisticEtaSec: eta,
            mergeProgressPct: mergeProgressPct(initialTokForEncode, ev.tokenCount),
          };
        }
        dashThrottle.schedule();
      },
    });
    dashThrottle.flush();

    draft.encodeProgress = null;
    corpusUtf16Units = corpus.length;

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
  const schedule = {
    lrMax: optim.lrMax,
    lrMin: optim.lrMin,
    warmupSteps: Math.min(10, Math.max(1, Math.floor(totalSteps / 10))),
    totalSteps,
  };

  draft.phase = 'trainer_init';
  draft.detailLine = `AdamW · lr ${String(optim.lrMax)}→${String(optim.lrMin)} · wd ${String(optim.weightDecay)} · clip ${Number.isFinite(optim.maxGradNorm) ? String(optim.maxGradNorm) : 'off'}`;
  dashThrottle.flush();
  const trainer = new Trainer(rt, model, reg, opt, {
    maxGradNorm: optim.maxGradNorm,
    schedule,
  });

  draft.archSummary = `${String(gptCfg.nLayers)}L · d${String(model.config.dModel)} · V${String(model.config.vocabSize)} · ${computeDtype}`;
  draft.phase = 'training';
  draft.trainingPhaseStartedAtMs = Date.now();
  draft.trainLrDisplay = schedule.lrMax;
  draft.detailLine = 'optimizer stepping (GPU shader warmup may pause step 1)';
  dashThrottle.flush();

  console.error(
    `train: optim lr_max=${String(optim.lrMax)} lr_min=${String(optim.lrMin)} weight_decay=${String(optim.weightDecay)} max_grad_norm=${Number.isFinite(optim.maxGradNorm) ? String(optim.maxGradNorm) : 'off'}`,
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
