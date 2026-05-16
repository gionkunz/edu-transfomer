import { BPETokenizer } from '@llm-transformer-diy/tokenizer';

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
