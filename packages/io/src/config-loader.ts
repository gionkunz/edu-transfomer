import { readFile } from 'node:fs/promises';

/** Parse a UTF-8 JSON file into an `unknown` value (callers narrow with validators). */
export async function readJsonFile(jsonPath: string): Promise<unknown> {
  const text = await readFile(jsonPath, 'utf8');
  return JSON.parse(text) as unknown;
}
