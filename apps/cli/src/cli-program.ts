import { Command } from 'commander';

/* eslint-disable @typescript-eslint/no-unsafe-member-access -- Commander option bags are untyped */

import { exitCliError } from './cli-errors.js';
import { runGenerateCommand } from './cmd/generate.js';
import { runInspect } from './cmd/inspect.js';
import { runTokenizerDecode, runTokenizerEncode, runTokenizerTrain } from './cmd/tokenizer.js';
import { runTrainCommand } from './cmd/train.js';

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
    .action(async (opts) => {
      await runTokenizerTrain({
        dataGlobs: opts.data as string[],
        vocabSize: opts.vocabSize as number,
        outPath: opts.out as string,
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
    .action(async (opts) => {
      await runTrainCommand({
        dataGlobs: opts.data as string[],
        outDir: opts.out as string,
        tokenizerPath: opts.tokenizer as string,
        configArg: opts.config as string,
        steps: Number(opts.steps),
        seqLen: Number(opts.seqLen),
      });
    });

  root
    .command('generate')
    .description('autoregressive generation with KV cache (WebGPU)')
    .requiredOption('--model <path>', 'checkpoint dir or model.json sidecar')
    .requiredOption('--tokenizer <path>', 'BPE tokenizer json')
    .requiredOption('--prompt <text>', 'prompt string')
    .option('--max-new-tokens <n>', 'tokens to sample after prompt', '64')
    .option('--temperature <x>', 'sampling temperature', '0.8')
    .option('--top-k <k>', 'top-k sampling (0 = off)', '40')
    .action(async (opts) => {
      await runGenerateCommand({
        modelArg: opts.model as string,
        tokenizerPath: opts.tokenizer as string,
        prompt: opts.prompt as string,
        maxNewTokens: Number(opts.maxNewTokens),
        temperature: Number(opts.temperature),
        topK: Number(opts.topK),
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
