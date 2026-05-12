import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';

import { discoverTrainingFiles } from '@llm-transformer-diy/io';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';

/** Train BPE on text from **`--data`** (glob) and write **`--out`**. */
export async function runTokenizerTrain(opts: {
  readonly dataGlobs: string[];
  readonly vocabSize: number;
  readonly outPath: string;
}): Promise<void> {
  const files = await discoverTrainingFiles(opts.dataGlobs);
  if (files.length === 0) {
    throw new Error('tokenizer train: no files matched --data');
  }
  const parts: string[] = [];
  for (const rel of files) {
    const abs = path.resolve(rel);
    parts.push(await readFile(abs, 'utf8'));
  }
  const corpus = parts.join('\n');
  const tok = new BPETokenizer();
  console.error(`Training BPE: ${String(files.length)} file(s), ${String(corpus.length)} chars…`);
  tok.train(corpus, opts.vocabSize);
  await mkdir(path.dirname(path.resolve(opts.outPath)), { recursive: true });
  await tok.save(opts.outPath);
  console.error(`Wrote tokenizer: ${opts.outPath}`);
}

export async function runTokenizerEncode(opts: {
  readonly tokenizerPath: string;
  readonly text: string;
}): Promise<void> {
  const tok = new BPETokenizer();
  await tok.load(opts.tokenizerPath);
  const ids = tok.encode(opts.text);
  console.log(ids.join(','));
}

export async function runTokenizerDecode(opts: {
  readonly tokenizerPath: string;
  readonly idsCsv: string;
}): Promise<void> {
  const tok = new BPETokenizer();
  await tok.load(opts.tokenizerPath);
  const parts = opts.idsCsv
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const ids = parts.map((p) => {
    const n = Number(p);
    if (!Number.isInteger(n)) {
      throw new Error(`decode: invalid id "${p}"`);
    }
    return n;
  });
  const text = tok.decode(ids);
  console.log(text);
}
