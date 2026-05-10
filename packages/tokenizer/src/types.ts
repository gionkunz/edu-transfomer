/**
 * Contract from `docs/project.md` — shared by char-level and future BPE implementations.
 */
export interface Tokenizer {
  train(text: string, vocabSize: number): void;
  encode(text: string): number[];
  decode(ids: number[]): string;
  save(path: string): Promise<void>;
  load(path: string): Promise<void>;
  readonly vocabSize: number;
}
