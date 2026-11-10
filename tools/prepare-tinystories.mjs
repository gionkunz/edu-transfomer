#!/usr/bin/env node
/**
 * Download TinyStories (V2, GPT-4 generated) and cut the training file into
 * shards that `llm train --data` can read.
 *
 *   node tools/prepare-tinystories.mjs [outDir] [shardMiB]
 *
 * Why shards: `llm train` reads each corpus file into one JS string, and V8
 * caps a string at ~512 MiB. The raw training file is 2.2 GB. Shards are cut
 * on story boundaries (`<|endoftext|>` lines), so no story is split.
 *
 * Output (default `corpus/tinystories`):
 *   raw/TinyStoriesV2-GPT4-{train,valid}.txt   downloads (skipped if present)
 *   train-000.txt … train-NNN.txt              ~shardMiB each (default 100)
 *   valid.txt                                  held-out stories for --val-data
 *
 * Dataset: Eldan & Li, "TinyStories: How Small Can Language Models Be and
 * Still Speak Coherent English?" (2023). License: CDLA-Sharing-1.0.
 */
import { createReadStream, createWriteStream, existsSync } from 'node:fs';
import { copyFile, mkdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const BASE = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main';
const TRAIN = 'TinyStoriesV2-GPT4-train.txt';
const VALID = 'TinyStoriesV2-GPT4-valid.txt';
const DELIMITER = Buffer.from('<|endoftext|>\n');

const outDir = path.resolve(process.argv[2] ?? 'corpus/tinystories');
const shardBytes = Number(process.argv[3] ?? '100') * 1024 * 1024;
if (!Number.isFinite(shardBytes) || shardBytes <= 0) {
  throw new Error('shardMiB must be a positive number');
}
const rawDir = path.join(outDir, 'raw');

async function download(name) {
  const dest = path.join(rawDir, name);
  if (existsSync(dest)) {
    console.error(`have ${dest} (${String((await stat(dest)).size)} bytes)`);
    return dest;
  }
  console.error(`downloading ${name} …`);
  const res = await fetch(`${BASE}/${name}`);
  if (!res.ok || res.body === null) {
    throw new Error(`download failed: ${name} → HTTP ${String(res.status)}`);
  }
  await pipeline(Readable.fromWeb(res.body), createWriteStream(`${dest}.part`));
  const { rename } = await import('node:fs/promises');
  await rename(`${dest}.part`, dest);
  return dest;
}

function shardPath(index) {
  return path.join(outDir, `train-${String(index).padStart(3, '0')}.txt`);
}

/** Stream `src` into shards, switching file at the first story boundary past `shardBytes`. */
async function shard(src) {
  let index = 0;
  let written = 0;
  let out = createWriteStream(shardPath(index));
  /** Bytes held back because they may be the start of a delimiter split across chunks. */
  let carry = Buffer.alloc(0);

  const write = (buf) =>
    new Promise((resolve) => {
      written += buf.length;
      if (out.write(buf)) resolve();
      else out.once('drain', resolve);
    });
  const rotate = async () => {
    await new Promise((resolve) => out.end(resolve));
    index += 1;
    written = 0;
    out = createWriteStream(shardPath(index));
  };

  for await (const chunk of createReadStream(src)) {
    let buf = carry.length > 0 ? Buffer.concat([carry, chunk]) : chunk;
    carry = Buffer.alloc(0);
    while (buf.length > 0) {
      if (written < shardBytes) {
        const take = Math.min(buf.length, shardBytes - written);
        await write(buf.subarray(0, take));
        buf = buf.subarray(take);
        continue;
      }
      // Shard is full: finish the current story, then rotate.
      const at = buf.indexOf(DELIMITER);
      if (at === -1) {
        // Keep a possible partial delimiter for the next chunk.
        const keep = Math.min(buf.length, DELIMITER.length - 1);
        await write(buf.subarray(0, buf.length - keep));
        carry = buf.subarray(buf.length - keep);
        buf = Buffer.alloc(0);
      } else {
        const end = at + DELIMITER.length;
        await write(buf.subarray(0, end));
        buf = buf.subarray(end);
        await rotate();
      }
    }
  }
  if (carry.length > 0) await write(carry);
  await new Promise((resolve) => out.end(resolve));
  return index + 1;
}

await mkdir(rawDir, { recursive: true });
const trainRaw = await download(TRAIN);
const validRaw = await download(VALID);
console.error('sharding training file …');
const shards = await shard(trainRaw);
await copyFile(validRaw, path.join(outDir, 'valid.txt'));
console.error(`wrote ${String(shards)} training shard(s) + valid.txt in ${outDir}`);
