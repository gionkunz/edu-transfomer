import { Command } from 'commander';

import { exitCliError } from './cli-errors';
import { runChatCommand } from './cmd/chat';
import {
  runEmbeddingsDecode,
  runEmbeddingsEncode,
  runEmbeddingsExpression,
  runEmbeddingsNearest,
  runEmbeddingsSimilarity,
} from './cmd/embeddings';
import { runGenerateCommand } from './cmd/generate';
import { DEFAULT_STOP_TEXT } from './cmd/generate-stop';
import { runInspect } from './cmd/inspect';
import { DEFAULT_LEARN_PROMPT, runLearnCommand } from './cmd/learn';
import { runTokenizerDecode, runTokenizerEncode } from './cmd/tokenizer';
import { runTokenizerTrain } from './cmd/tokenizer-train';
import {
  runTrainCommand,
  TRAIN_DEFAULT_ACCUM,
  TRAIN_DEFAULT_CHECKPOINT_EVERY,
  TRAIN_DEFAULT_EVAL_BATCHES,
  TRAIN_DEFAULT_EVAL_EVERY,
  TRAIN_DEFAULT_LR_MAX,
  TRAIN_DEFAULT_LR_MIN,
  TRAIN_DEFAULT_MAX_GRAD_NORM,
  TRAIN_DEFAULT_WARMUP_FRACTION,
  TRAIN_DEFAULT_WEIGHT_DECAY,
} from './cmd/train';
import { parseOptionalComputeDtype } from './compute-dtype';
import { resolveTokenizerPath } from './paths';

function tokenizerFromModelOpts(
  opts: { model: unknown; tokenizer?: unknown },
  commandLabel: string,
): string {
  try {
    return resolveTokenizerPath(opts.model as string, opts.tokenizer as string | undefined);
  } catch (e) {
    if (e instanceof Error) {
      throw new Error(`${commandLabel}: ${e.message}`);
    }
    throw e;
  }
}

function parseTrainFiniteOption(raw: unknown, flag: string): number {
  const v = Number(raw);
  if (!Number.isFinite(v)) {
    throw new Error(`llm train: ${flag} must be a finite number (got ${String(raw)})`);
  }
  return v;
}

/** Accepts finite norm, **0** (no clip), or **`Infinity`** (literal string `Infinity`). */
function parseMaxGradNormOption(raw: unknown): number {
  if (typeof raw === 'string' && raw.trim().toLowerCase() === 'infinity') {
    return Number.POSITIVE_INFINITY;
  }
  const v = Number(raw);
  if (v === 0) {
    return 0;
  }
  if (v === Number.POSITIVE_INFINITY) {
    return Number.POSITIVE_INFINITY;
  }
  if (!Number.isFinite(v) || v < 0) {
    throw new Error(
      `llm train: --max-grad-norm must be ≥ 0, Infinity, or 0 to disable (got ${String(raw)})`,
    );
  }
  return v;
}

function parseReplayRows(raw: unknown): number[] | undefined {
  if (raw === undefined || raw === null || String(raw).trim() === '') {
    return undefined;
  }
  const parts = String(raw).split(',');
  const rows: number[] = [];
  for (const p of parts) {
    const v = Number(p.trim());
    if (!Number.isInteger(v) || v < 0) {
      throw new Error(
        `llm train: --replay-rows must be comma-separated non-negative integers (got ${String(raw)})`,
      );
    }
    rows.push(v);
  }
  return rows;
}

function collectStrings(value: string, prev: string[]): string[] {
  return prev.concat(value);
}

export function createCliProgram(): Command {
  const root = new Command('llm');
  root
    .description('Transformer toolkit CLI')
    .showHelpAfterError(true)
    .option('-v, --verbose', 'print full error stacks');

  const tok = root.command('tokenizer').description('Train or run BPE');

  tok
    .command('train')
    .description('train a BPE tokenizer on text files')
    .requiredOption(
      '--data <glob>',
      'glob for training files (repeatable)',
      collectStrings,
      [] as string[],
    )
    .requiredOption('--vocab-size <n>', 'target vocab size', (v) => Number(v))
    .requiredOption('--out <path>', 'output tokenizer json path')
    .option(
      '--disk-stream',
      'stream corpus to a temp u32 file (very large data; avoids JS max string length / RAM limits; slower)',
      false,
    )
    .action(async (opts) => {
      await runTokenizerTrain({
        dataGlobs: opts.data as string[],
        vocabSize: opts.vocabSize as number,
        outPath: opts.out as string,
        diskStream: Boolean(opts.diskStream),
      });
    });

  tok
    .command('encode')
    .description('encode text to comma-separated token ids')
    .requiredOption('--tokenizer <path>', 'tokenizer json')
    .argument('<text...>', 'input text')
    .action(async (words: string[], opts) => {
      const text = words.join(' ');
      await runTokenizerEncode({ tokenizerPath: opts.tokenizer as string, text });
    });

  tok
    .command('decode')
    .description('decode comma-separated token ids to text')
    .requiredOption('--tokenizer <path>', 'tokenizer json')
    .argument('<ids>', 'comma-separated ids, e.g. 42,17,8')
    .action(async (ids: string, opts) => {
      await runTokenizerDecode({ tokenizerPath: opts.tokenizer as string, idsCsv: ids });
    });

  root
    .command('train')
    .description('fine-tune / train GPT (WebGPU)')
    .requiredOption(
      '--data <glob>',
      'training files glob (repeatable)',
      collectStrings,
      [] as string[],
    )
    .requiredOption('--out <dir>', 'output directory for checkpoint')
    .requiredOption('--tokenizer <path>', 'BPE tokenizer json')
    .requiredOption('--config <name|file>', 'preset (tiny, small) or GptConfig json path')
    .option('--steps <n>', 'optimizer steps', '100')
    .option('--seq-len <n>', 'LM context length', '32')
    .option(
      '--token-cache <base>',
      'cache joined-corpus token ids at <base>.meta.json + little-endian <base>.ids (skip read+BPE on cache hit)',
    )
    .option(
      '--rebuild-token-cache',
      're-encode and overwrite token cache (requires --token-cache)',
      false,
    )
    .option(
      '--lr-max <x>',
      `peak learning rate after warmup (cosine schedule; default ${String(TRAIN_DEFAULT_LR_MAX)})`,
      String(TRAIN_DEFAULT_LR_MAX),
    )
    .option(
      '--lr-min <x>',
      `cosine schedule floor LR (default ${String(TRAIN_DEFAULT_LR_MIN)})`,
      String(TRAIN_DEFAULT_LR_MIN),
    )
    .option(
      '--weight-decay <x>',
      `AdamW decoupled weight decay λ (default ${String(TRAIN_DEFAULT_WEIGHT_DECAY)})`,
      String(TRAIN_DEFAULT_WEIGHT_DECAY),
    )
    .option(
      '--max-grad-norm <x>',
      `global L2 grad clip (0 or Infinity = off; default ${String(TRAIN_DEFAULT_MAX_GRAD_NORM)})`,
      String(TRAIN_DEFAULT_MAX_GRAD_NORM),
    )
    .option(
      '--warmup <x>',
      `linear LR warmup as a fraction of --steps (0 < x < 1; default ${String(TRAIN_DEFAULT_WARMUP_FRACTION)})`,
      String(TRAIN_DEFAULT_WARMUP_FRACTION),
    )
    .option(
      '--checkpoint-every <n>',
      `save model + optimizer every N steps (0 = final only; default ${String(TRAIN_DEFAULT_CHECKPOINT_EVERY)})`,
      String(TRAIN_DEFAULT_CHECKPOINT_EVERY),
    )
    .option(
      '--resume',
      'continue from model.bin + train-state.json in --out (same config/steps/seq-len/dtype required)',
      false,
    )
    .option(
      '--replay-step <n>',
      'with --resume: fast-forward to step n and run a forward layer probe (use with --replay-only)',
    )
    .option('--replay-only', 'exit after --replay-step probe instead of continuing training', false)
    .option(
      '--replay-rows <i,...>',
      'sequence row indices for replay probe (default: 0,mid,last; e.g. 229 or 229,283)',
    )
    .option('--dtype <d>', 'parameter compute dtype during training: f32 | bf16', 'f32')
    .option(
      '--accum <n>',
      `sequences per optimizer step, gradients averaged (default ${String(TRAIN_DEFAULT_ACCUM)})`,
      String(TRAIN_DEFAULT_ACCUM),
    )
    .option(
      '--val-data <glob>',
      'held-out files glob for validation loss (repeatable; must not overlap --data)',
      collectStrings,
      [] as string[],
    )
    .option(
      '--eval-every <n>',
      `with --val-data: measure validation loss every N steps (default ${String(TRAIN_DEFAULT_EVAL_EVERY)})`,
      String(TRAIN_DEFAULT_EVAL_EVERY),
    )
    .option(
      '--eval-batches <n>',
      `with --val-data: held-out sequences per measurement (default ${String(TRAIN_DEFAULT_EVAL_BATCHES)})`,
      String(TRAIN_DEFAULT_EVAL_BATCHES),
    )
    .action(async (opts) => {
      const tokenCacheRaw = opts.tokenCache as string | undefined;
      const tokenCacheBase =
        tokenCacheRaw !== undefined && String(tokenCacheRaw).trim() !== ''
          ? String(tokenCacheRaw).trim()
          : undefined;
      const rebuildTokenCache = Boolean(opts.rebuildTokenCache);
      if (rebuildTokenCache && !tokenCacheBase) {
        throw new Error('train: --rebuild-token-cache requires --token-cache <base>');
      }
      const computeDtype = parseOptionalComputeDtype(opts.dtype as string | undefined, 'train');
      const replayRows = parseReplayRows(opts.replayRows);
      await runTrainCommand({
        dataGlobs: opts.data as string[],
        outDir: opts.out as string,
        tokenizerPath: opts.tokenizer as string,
        configArg: opts.config as string,
        steps: Number(opts.steps),
        seqLen: Number(opts.seqLen),
        lrMax: parseTrainFiniteOption(opts.lrMax, '--lr-max'),
        lrMin: parseTrainFiniteOption(opts.lrMin, '--lr-min'),
        weightDecay: parseTrainFiniteOption(opts.weightDecay, '--weight-decay'),
        maxGradNorm: parseMaxGradNormOption(opts.maxGradNorm),
        warmupFraction: parseTrainFiniteOption(opts.warmup, '--warmup'),
        checkpointEvery: Number(opts.checkpointEvery),
        computeDtype,
        accum: Number(opts.accum),
        valGlobs: opts.valData as string[],
        evalEvery: Number(opts.evalEvery),
        evalBatches: Number(opts.evalBatches),
        ...(Boolean(opts.resume) ? { resume: true } : {}),
        ...(opts.replayStep !== undefined && String(opts.replayStep).trim() !== ''
          ? { replayStep: Number(opts.replayStep) }
          : {}),
        ...(Boolean(opts.replayOnly) ? { replayOnly: true } : {}),
        ...(replayRows !== undefined ? { replayRows } : {}),
        ...(tokenCacheBase !== undefined ? { tokenCacheBase } : {}),
        ...(rebuildTokenCache ? { rebuildTokenCache: true } : {}),
      });
    });

  root
    .command('generate')
    .description('autoregressive generation with KV cache (WebGPU)')
    .requiredOption('--model <path>', 'checkpoint dir or model.json sidecar')
    .option('--tokenizer <path>', 'BPE tokenizer json (default: tokenizer.json in checkpoint dir)')
    .requiredOption('--prompt <text>', 'prompt string')
    .option('--max-new-tokens <n>', 'tokens to sample after prompt', '64')
    .option('--temperature <x>', 'sampling temperature', '0.8')
    .option('--top-k <k>', 'top-k sampling (0 = off)', '40')
    .option('--dtype <d>', 'parameter compute dtype for the loaded model: f32 | bf16', 'f32')
    .option(
      '--allow-dtype-upcast',
      'allow checkpoints whose manifest dtypes differ from --dtype (16-bit → f32; OpenSpec §8)',
    )
    .option(
      '--repeat-penalty <p>',
      'down-weight logits for recent tokens (GPT-2 style; 1 = off)',
      '1',
    )
    .option('--repeat-window <n>', 'tail context length for --repeat-penalty', '64')
    .option('--stop <text>', 'end generation when the model writes this marker', DEFAULT_STOP_TEXT)
    .option('--no-stop', 'keep generating past the end-of-text marker, up to --max-new-tokens')
    .option(
      '--no-explore',
      'disable the educational explore TUI (probability inspector, attention panel, architecture view, hotkeys: s slow, p pause, up/down block, f architecture); use the minimal streaming view instead. Explore needs a terminal of at least 60×24; the full attention / architecture panels (one row per head, one to two per block) need a taller one and compress or hide below that.',
    )
    .action(async (opts) => {
      const repeatPenalty = Number(opts.repeatPenalty as string);
      const repeatWindow = Number(opts.repeatWindow as string);
      if (!Number.isFinite(repeatPenalty) || repeatPenalty < 1) {
        throw new Error('generate: --repeat-penalty must be a finite number ≥ 1');
      }
      if (!Number.isInteger(repeatWindow) || repeatWindow < 1) {
        throw new Error('generate: --repeat-window must be a positive integer');
      }
      const computeDtype = parseOptionalComputeDtype(opts.dtype as string | undefined, 'generate');
      await runGenerateCommand({
        modelArg: opts.model as string,
        tokenizerPath: tokenizerFromModelOpts(opts, 'generate'),
        prompt: opts.prompt as string,
        maxNewTokens: Number(opts.maxNewTokens),
        temperature: Number(opts.temperature),
        topK: Number(opts.topK),
        explore: opts.explore !== false,
        ...(typeof opts.stop === 'string' && opts.stop.length > 0 ? { stopText: opts.stop } : {}),
        computeDtype,
        ...(opts.allowDtypeUpcast === true ? { allowDtypeUpcast: true } : {}),
        ...(repeatPenalty > 1
          ? { repetitionPenalty: repeatPenalty, repetitionWindow: repeatWindow }
          : {}),
      });
    });

  root
    .command('learn')
    .description(
      'guided lesson: how the model predicts the next token, step by step (WebGPU, interactive)',
    )
    .argument('[prompt...]', 'prompt whose next token is explained', [])
    .requiredOption('--model <path>', 'checkpoint dir or model.json sidecar')
    .option('--tokenizer <path>', 'BPE tokenizer json (default: tokenizer.json in checkpoint dir)')
    .option('--temperature <x>', 'sampling temperature the lesson starts with', '0.8')
    .option('--top-k <k>', 'top-k the lesson starts with (0 = off)', '40')
    .option('--dtype <d>', 'parameter compute dtype for the loaded model: f32 | bf16', 'f32')
    .option(
      '--allow-dtype-upcast',
      'allow checkpoints whose manifest dtypes differ from --dtype (16-bit → f32; OpenSpec §8)',
    )
    .option(
      '--no-animation',
      'show every step in its finished state instead of building it up. Keys: left/right step, up/down block or temperature, h/H head, k top-k, r redraw, n continue with the drawn token, space replay, q quit. Needs a terminal of at least 80×24.',
    )
    .action(async (promptWords: string[], opts) => {
      const temperature = Number(opts.temperature as string);
      const topK = Number(opts.topK as string);
      if (!Number.isFinite(temperature) || temperature < 0) {
        throw new Error('learn: --temperature must be a finite number ≥ 0');
      }
      if (!Number.isInteger(topK) || topK < 0) {
        throw new Error('learn: --top-k must be a non-negative integer');
      }
      const computeDtype = parseOptionalComputeDtype(opts.dtype as string | undefined, 'learn');
      const prompt = promptWords.join(' ');
      await runLearnCommand({
        modelArg: opts.model as string,
        ...(typeof opts.tokenizer === 'string' ? { tokenizerArg: opts.tokenizer } : {}),
        prompt: prompt.length > 0 ? prompt : DEFAULT_LEARN_PROMPT,
        temperature,
        topK,
        animation: opts.animation !== false,
        computeDtype,
        ...(opts.allowDtypeUpcast === true ? { allowDtypeUpcast: true } : {}),
      });
    });

  root
    .command('chat')
    .description('interactive chat TUI with KV-backed generation (WebGPU)')
    .requiredOption('--model <path>', 'checkpoint dir or model.json sidecar')
    .option('--tokenizer <path>', 'BPE tokenizer json (default: tokenizer.json in checkpoint dir)')
    .option('--max-new-tokens <n>', 'tokens to sample per assistant reply', '64')
    .option('--temperature <x>', 'sampling temperature', '0.8')
    .option('--top-k <k>', 'top-k sampling (0 = off)', '40')
    .option('--dtype <d>', 'parameter compute dtype for the loaded model: f32 | bf16', 'f32')
    .option(
      '--allow-dtype-upcast',
      'allow checkpoints whose manifest dtypes differ from --dtype (16-bit → f32; OpenSpec §8)',
    )
    .option(
      '--repeat-penalty <p>',
      'down-weight logits for recent tokens (GPT-2 style; 1 = off)',
      '1',
    )
    .option('--repeat-window <n>', 'tail context length for --repeat-penalty', '64')
    .action(async (opts) => {
      const repeatPenalty = Number(opts.repeatPenalty as string);
      const repeatWindow = Number(opts.repeatWindow as string);
      if (!Number.isFinite(repeatPenalty) || repeatPenalty < 1) {
        throw new Error('chat: --repeat-penalty must be a finite number ≥ 1');
      }
      if (!Number.isInteger(repeatWindow) || repeatWindow < 1) {
        throw new Error('chat: --repeat-window must be a positive integer');
      }
      const computeDtype = parseOptionalComputeDtype(opts.dtype as string | undefined, 'chat');
      await runChatCommand({
        modelArg: opts.model as string,
        tokenizerPath: tokenizerFromModelOpts(opts, 'chat'),
        maxNewTokens: Number(opts.maxNewTokens),
        temperature: Number(opts.temperature),
        topK: Number(opts.topK),
        computeDtype,
        ...(opts.allowDtypeUpcast === true ? { allowDtypeUpcast: true } : {}),
        ...(repeatPenalty > 1
          ? { repetitionPenalty: repeatPenalty, repetitionWindow: repeatWindow }
          : {}),
      });
    });

  const emb = root
    .command('embeddings')
    .description(
      'explore the trained token-embedding geometry (encode/decode/similarity/expression/nearest)',
    );

  const addCheckpointOpts = (c: Command): Command =>
    c
      .requiredOption('--model <path>', 'checkpoint dir or model.json sidecar')
      .option(
        '--tokenizer <path>',
        'BPE tokenizer json (default: tokenizer.json in checkpoint dir)',
      )
      .option('--json', 'emit machine-readable JSON instead of table/text', false)
      .option(
        '--no-auto-space',
        'do NOT prepend a leading space to bare word inputs (BPE pretokenizers usually keep " king" and "king" as different tokens; auto-space defaults to ON so words encode like they appear in real text)',
      );

  const parseTopK = (raw: unknown): number => {
    const k = Number(raw);
    if (!Number.isInteger(k) || k <= 0) {
      throw new Error(`embeddings: --top-k must be a positive integer (got ${String(raw)})`);
    }
    return k;
  };

  const readAutoSpace = (opts: { autoSpace?: unknown }): boolean => opts.autoSpace !== false;

  addCheckpointOpts(
    emb
      .command('encode')
      .description('look up the token-embedding row(s) for the given text (default: mean-pooled)')
      .argument('<text...>', 'input text (joined with spaces)')
      .option('--per-token', 'emit one vector per token instead of mean-pooling', false),
  ).action(async (textParts: string[], opts) => {
    await runEmbeddingsEncode({
      modelArg: opts.model as string,
      tokenizerPath: tokenizerFromModelOpts(opts, 'embeddings encode'),
      text: textParts.join(' '),
      perToken: Boolean(opts.perToken),
      json: Boolean(opts.json),
      autoSpace: readAutoSpace(opts),
    });
  });

  addCheckpointOpts(
    emb
      .command('decode')
      .description('top-K vocabulary tokens nearest a vector (word, phrase, or @file.json)')
      .argument('<vector-spec>', 'word/phrase to encode, or @path/to/vec.json')
      .option('--top-k <n>', 'number of results to return', '10'),
  ).action(async (vectorSpec: string, opts) => {
    await runEmbeddingsDecode({
      modelArg: opts.model as string,
      tokenizerPath: tokenizerFromModelOpts(opts, 'embeddings decode'),
      vectorSpec,
      topK: parseTopK(opts.topK),
      json: Boolean(opts.json),
      autoSpace: readAutoSpace(opts),
    });
  });

  addCheckpointOpts(
    emb
      .command('similarity')
      .description('cosine similarity between two embeddings (each may be a word/phrase or @file)')
      .argument('<a>', 'first input (word/phrase or @file)')
      .argument('<b>', 'second input (word/phrase or @file)'),
  ).action(async (a: string, b: string, opts) => {
    await runEmbeddingsSimilarity({
      modelArg: opts.model as string,
      tokenizerPath: tokenizerFromModelOpts(opts, 'embeddings similarity'),
      a,
      b,
      json: Boolean(opts.json),
      autoSpace: readAutoSpace(opts),
    });
  });

  addCheckpointOpts(
    emb
      .command('expression')
      .description(
        'evaluate an arithmetic expression over embeddings (e.g. "king - man + woman") and return top-K nearest tokens',
      )
      .argument('<expr>', 'expression string')
      .option('--top-k <n>', 'number of results to return', '10')
      .option(
        '--exclude-inputs [bool]',
        'exclude expression input tokens from results (default true)',
        'true',
      ),
  ).action(async (expression: string, opts) => {
    const raw = String(opts.excludeInputs ?? 'true').toLowerCase();
    const excludeInputs = !(raw === 'false' || raw === '0' || raw === 'no');
    await runEmbeddingsExpression({
      modelArg: opts.model as string,
      tokenizerPath: tokenizerFromModelOpts(opts, 'embeddings expression'),
      expression,
      topK: parseTopK(opts.topK),
      excludeInputs,
      json: Boolean(opts.json),
      autoSpace: readAutoSpace(opts),
    });
  });

  addCheckpointOpts(
    emb
      .command('nearest')
      .description('top-K vocabulary tokens nearest the given word (alias for `decode <word>`)')
      .argument('<word>', 'input word or phrase')
      .option('--top-k <n>', 'number of results to return', '10'),
  ).action(async (word: string, opts) => {
    await runEmbeddingsNearest({
      modelArg: opts.model as string,
      tokenizerPath: tokenizerFromModelOpts(opts, 'embeddings nearest'),
      word,
      topK: parseTopK(opts.topK),
      json: Boolean(opts.json),
      autoSpace: readAutoSpace(opts),
    });
  });

  root
    .command('inspect')
    .description('print checkpoint config and tensor manifest summary')
    .requiredOption('--model <path>', 'checkpoint dir or model.json sidecar')
    .action(async (opts) => {
      await runInspect(opts.model as string);
    });

  return root;
}

export async function runCli(argv: string[]): Promise<void> {
  const verbose = argv.includes('--verbose') || argv.includes('-v');
  const program = createCliProgram();
  try {
    await program.parseAsync(argv);
  } catch (e) {
    exitCliError(e, verbose);
  }
}
