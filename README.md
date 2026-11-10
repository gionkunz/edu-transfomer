# llm-transformer-diy

Educational **GPT-style transformer** stack in **TypeScript**: custom tensors, **WebGPU/WGSL** kernels, **autograd**, GPT blocks, training, inference, and an **`llm`** CLI—with **no** heavyweight ML frameworks in the compute path.

Canonical architecture, dependency rules, and layout live in **`docs/project.md`**. Conceptual narration: **`docs/transformer-explained.md`**.

---

## Stack (mental model)

Layering is strict bottom‑up:

`gpu` → `tensor` → `autograd` → `model` → `training` / `inference` → **`apps/cli`**

The **`llm`** CLI only **wires** tokenizer, checkpoints, GPU init, wolf-tui, and **`generate`/`train`/`chat`** loops; modeling and kernels live in packages below it.

---

## Prerequisites

| Requirement                                | Notes                                                                                                                                                                                               |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Node.js**                                | **≥ 20** (see root `package.json` `engines`)                                                                                                                                                        |
| **pnpm**                                   | Workspace uses `pnpm` (install from [pnpm.io](https://pnpm.io))                                                                                                                                     |
| **WebGPU for `train`, `generate`, `chat`** | Use a GPU-capable **Node + Dawn/WebGPU** setup. Training fails loudly without a usable adapter; LLVMpipe/software adapters are painfully slow—see **`llm train`** stderr hints on Linux Vulkan/ICD. |

---

## Install and expose `llm` on your PATH

From the repo root:

```bash
pnpm install
pnpm exec nx run @llm-transformer-diy/cli:build
pnpm run cli:link-global
```

That builds **`dist/apps/cli`**, installs its pruned deps, runs the bootstrap **`main.js`** shebang step, and **`npm link`**s the **`@llm-transformer-diy/cli`** package so the **`llm`** binary is available globally (same as publishing/installing this CLI package elsewhere).

Smoke test:

```bash
llm --help
```

To remove the global link:

```bash
pnpm run cli:unlink-global
```

---

## CLI overview

After linking, **`llm`** exposes (among others):

| Command                               | Role                                                                                                                                                                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`llm tokenizer train`**             | Fit **BPE** on text files matched by **`--data` globs**; write tokenizer JSON                                                                                                                                                 |
| **`llm tokenizer encode` / `decode`** | Tiny encode/decode utilities (stdout; no fullscreen TUI)                                                                                                                                                                      |
| **`llm train`**                       | Load corpus + tokenizer → **WebGPU** training → checkpoint dir (**wolf-tui** dashboard, including corpus **BPE encode** progress where applicable). Optional **`--dtype f32\|bf16`** (default **`f32`**) for compute weights. |
| **`llm generate`**                    | Single prompt, streaming **wolf-tui**; **`--dtype`** and optional **`--allow-dtype-upcast`** for checkpoint vs runtime dtype.                                                                                                 |
| **`llm chat`**                        | Multi-turn **wolf-tui** (same sampling flags flavour as **`generate`)**; same dtype flags as **`generate`**.                                                                                                                  |
| **`llm inspect`**                     | Checkpoint / **`model.json`** summary on stdout                                                                                                                                                                               |
| **`llm embeddings`**                  | Explore the trained token-embedding geometry: `encode` / `decode` / `nearest` / `similarity` / `expression` (vector arithmetic with a tiny `+ - ( )` language). Stdout only; no fullscreen TUI.                               |

For full flags, run **`llm train --help`**, **`llm tokenizer train --help`**, **`llm generate --help`**, **`llm chat --help`**, etc. Training and inference share the **`--dtype`** knob (`f32` / `bf16`); **`generate`** / **`chat`** also expose **`--allow-dtype-upcast`** when loading **`bf16`** checkpoints into an **`f32`** model—see **`docs/project.md`** (checkpoint + CLI sections).

Built-in GPT shape **presets** for **`llm train --config`** (vocab comes from tokenizer only): **`tiny`**, **`small`**. Otherwise pass a **`GptConfig` JSON path**—see **`config/tinystories.json`**, **`config/xxs.json`**, **`config/xs.json`** and **`config/s.json`** in this repo (`vocabSize` in the file **must agree** with the tokenizer’s **`vocabSize`**).

---

## Quick start: a model that tells a tiny story

The fastest route to readable output is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories): 2.2 GB of short stories written with a small child's vocabulary, built so that models under ~10M parameters can learn coherent English. A handful of novels is far too little text for that (the model memorizes fragments instead of learning grammar).

```bash
# 1. Download + cut into ~100 MB shards (story boundaries) under corpus/tinystories/ (gitignored)
node tools/prepare-tinystories.mjs

# 2. Tokenizer: one shard is plenty for an 8k vocabulary (~1 min)
llm tokenizer train --data 'corpus/tinystories/train-000.txt' --vocab-size 8000 --out out/tinystories/tokenizer.json

# 3. Train. config/tinystories.json is ~10M parameters (8 layers, dModel 256, context 512).
llm train \
  --data 'corpus/tinystories/train-00*.txt' \
  --val-data corpus/tinystories/valid.txt \
  --tokenizer out/tinystories/tokenizer.json \
  --config config/tinystories.json \
  --out out/tinystories/run1 \
  --token-cache out/tinystories/cache-10shards \
  --seq-len 512 --accum 4 \
  --steps 60000 \
  --lr-max 1e-3 --lr-min 1e-4 --max-grad-norm 1 --warmup 0.02

# 4. Sample
llm generate --model out/tinystories/run1 --prompt "Once upon a time" --max-new-tokens 200
```

- **`--data 'train-00*.txt'`** is the first 10 shards (~1 GB, roughly 250M tokens). Use `train-*.txt` for everything; the token ids are held in memory (4 bytes per token).
- **`--accum 4`** runs 4 sequences per optimizer step and averages their gradients — a batch of 4 at the memory cost of one. Tokens per step = `seq-len × accum`.
- **`--val-data`** adds a held-out loss curve (every `--eval-every` steps). Train loss falling while validation loss rises means memorization.
- Every step is appended to **`<out>/metrics.jsonl`** (`step`, `loss`, `lr`, `gradNorm`, `tokPerSec`, `valLoss`), and `--resume` re-plots it.
- Stories in the corpus are separated by a literal `<|endoftext|>` line, which the tokenizer encodes as three ordinary tokens; generated text contains it where the model ends a story.

## Example: train a tokenizer

Pick a **`--vocab-size`** that matches the model JSON you’ll use (e.g. **`8000`** for **`config/tinystories.json`**, **`10000`** for **`config/xs.json`** / **`config/s.json`**). **`--data`** is repeatable; each value is a glob.

```bash
llm tokenizer train --data 'corpus/real/*.txt' --vocab-size 8000 --out out/tok-8000.json
```

For several disjoint globs:

```bash
llm tokenizer train --data 'corpus/real/*.txt' --data 'notes/*.txt' --vocab-size 8000 --out out/tok-8000.json
```

> **BPE pre-tokenizer.** Tokenizers are trained under a mandatory **`gpt2-v1`** pre-tokenizer (GPT-2 / Llama style: a content region MAY absorb a single leading ASCII space, English contractions form their own regions, trailing whitespace at end-of-line stays isolated). This roughly halves token count for English text vs. the older isolated-whitespace scheme and is the standard most production tokenizers use. Encode walks each region independently and reaches quiescence in ~5 outer passes per region regardless of corpus size, dropping per-corpus encode from "hours" to "seconds" on 100 MB+ inputs. **Tokenizer JSON files saved before this change** no longer load (schema `version: 3` with `pretokenizer: 'gpt2-v1'` is required) — retrain to migrate. See `docs/project.md` → _BPE Tokenizer_ for the policy details.

---

## Example: train a model

Minimal shape (adjust **`--steps`**, **`--seq-len`**, **`--out`**, and globs):

```bash
llm train --data 'corpus/real/pg*.txt' --tokenizer out/tok-8000.json --config config/medium-longctx.json --out out/medium-longctx --steps 50000 --seq-len 512 --token-cache out/cache/my-corpus
```

**Semantics worth remembering:**

- **`--seq-len`** — length of each **training slice** sampled from the tokenized corpus. Must be **`≤ maxSeqLen`** in the GPT config (**`medium.json`** uses **`512`**; **`medium-longctx.json`** uses **`1024`**—train with **`--seq-len 1024`** if you use that config).
- **`--token-cache`** — optional base path for a **joined-corpus token id cache** (skips giant re-encodes across runs); use **`--rebuild-token-cache`** with **`--token-cache`** to force rebuild.

Alternative **`--config tiny`** / **`small`** skips custom JSON presets.

---

## Example: push harder for lower loss, faster

The defaults are **conservative** so a fresh checkout converges _somewhere_ on most adapters. To actually drive loss down on a given corpus, escalate a few knobs together. Run **`llm train --help`** to see the full list and current defaults; the relevant ones are:

| Flag                  | Default | What it does                                                                                  |
| --------------------- | ------- | --------------------------------------------------------------------------------------------- |
| **`--steps`**         | `100`   | Optimizer steps. More steps = more loss progress (until you overfit).                         |
| **`--seq-len`**       | `32`    | Tokens per training slice → tokens of LM signal per step. Must be **`≤ maxSeqLen`** in cfg.   |
| **`--lr-max`**        | `1e-4`  | Peak LR after warmup (cosine schedule). Bigger = faster early progress, easier to diverge.    |
| **`--lr-min`**        | `1e-5`  | Cosine floor LR. Set to `~lrMax/10` for a real decay; raise to keep learning late in the run. |
| **`--weight-decay`**  | `0.01`  | AdamW decoupled L2 (regularization). `0` to disable; `0.05–0.1` for stronger regularization.  |
| **`--max-grad-norm`** | `0.5`   | Global grad-norm clip. Looser (`1.0`) lets bigger updates through; `0`/`Infinity` disables.   |
| **`--dtype`**         | `f32`   | `bf16` ≈ 2× throughput on capable adapters (more steps in the same wall time).                |
| **`--config`**        | —       | Bigger shape (more layers / wider `dModel`) raises the loss ceiling you can reach.            |
| **`--token-cache`**   | —       | Keep this between runs so you don’t re-BPE the corpus on every escalation.                    |

### Step 1 — bigger context, more steps, real LR decay

Compared to the earlier example: same `medium-longctx` shape, but **`--seq-len 1024`** (matches `maxSeqLen`), **5×** more steps, and a **10×** LR range so cosine actually decays.

```bash
llm train \
  --data 'corpus/real/pg*.txt' \
  --tokenizer out/tok-8000.json \
  --config config/medium-longctx.json \
  --out out/medium-longctx-push1 \
  --token-cache out/cache/my-corpus \
  --steps 250000 \
  --seq-len 1024 \
  --lr-max 3e-4 \
  --lr-min 3e-5
```

### Step 2 — push the optimizer + use bf16 for throughput

Larger peak LR and a looser grad clip mean more loss-per-step **when it doesn’t diverge**; **`--dtype bf16`** gives you roughly **2×** more steps per minute on capable adapters, which is usually the bigger win.

```bash
llm train \
  --data 'corpus/real/pg*.txt' \
  --tokenizer out/tok-8000.json \
  --config config/medium-longctx.json \
  --out out/medium-longctx-push2 \
  --token-cache out/cache/my-corpus \
  --steps 400000 \
  --seq-len 1024 \
  --lr-max 6e-4 \
  --lr-min 6e-5 \
  --weight-decay 0.05 \
  --max-grad-norm 1.0 \
  --dtype bf16
```

### Step 3 — raise the ceiling with a bigger model

When the loss curve flattens and **train loss is still high**, you’re capacity-bound, not optimizer-bound. Drop a deeper / wider config beside `config/medium-longctx.json` (same `vocabSize`, same `maxSeqLen` you train at) and reuse the same token cache:

```json
// config/large-longctx.json (must keep vocabSize = tokenizer vocab)
{ "vocabSize": 8000, "maxSeqLen": 1024, "dModel": 384, "nHeads": 12, "dFf": 1536, "nLayers": 6 }
```

```bash
llm train \
  --data 'corpus/real/pg*.txt' \
  --tokenizer out/tok-8000.json \
  --config config/large-longctx.json \
  --out out/large-longctx-push \
  --token-cache out/cache/my-corpus \
  --steps 400000 \
  --seq-len 1024 \
  --lr-max 4e-4 \
  --lr-min 4e-5 \
  --dtype bf16
```

### Caveats when pushing

- **Divergence**: if loss spikes to `NaN`/`Inf` or stops decreasing after a few hundred steps, **halve `--lr-max`** before changing anything else. Bigger models usually need a **smaller** peak LR than tiny ones.
- **`--dtype bf16`** halves memory and bandwidth versus **`f32`** while keeping **`f32`**'s exponent range, so it trains stably without loss scaling; **`fp16`** is intentionally not supported.
- **Disabling clipping** (`--max-grad-norm 0` or `Infinity`) is fine for short experiments but risky on long runs; the trainer message at startup tells you which mode is active.
- **`--seq-len` must be `≤ maxSeqLen`** in the GPT config, otherwise training fails fast on bounds.
- **Always pass `--token-cache <base>`** when iterating — re-encoding a real corpus can dwarf the actual training time on the first run.

---

## Example: inference

**One-shot:**

```bash
llm generate --model out/run-medium --tokenizer out/tok-8000.json --prompt "Once upon a time" --max-new-tokens 128
```

**Interactive chat** (wolf-tui; Ctrl+C exits):

```bash
llm chat --model out/run-medium --tokenizer out/tok-8000.json --max-new-tokens 96
```

Sampling flags (**`temperature`**, **`top-k`**, **`repeat-penalty`**, **`repeat-window`**) mirror **`generate`**; see **`llm chat --help`**.

---

## Example: explore the embedding space

A trained checkpoint contains a learned **token-embedding matrix** (`tokenEmbed`, shape `[vocab, dModel]`) — one vector per BPE token. **`llm embeddings`** lets you poke at that geometry: encode words to vectors, find nearest neighbours by cosine similarity, compare two words, and evaluate small arithmetic expressions like the classic _king − man + woman ≈ queen_. It’s a fast, qualitative read on what the model has actually learned.

All subcommands take a checkpoint (`--model`) and the matching tokenizer (`--tokenizer`), print to stdout, and accept **`--json`** for piping into `jq`. Math runs CPU-side; no GPU is required.

> **Reading the output: cosine + band + z.** Raw cosine values are hard to interpret because the embedding cone is narrow and per-checkpoint (e.g. cos = 0.15 may be a strong relation on one model and noise on another). Every result is therefore tagged with a **z-score** and a **qualitative band** (`identical` / `very similar` / `similar` / `weakly similar` / `unrelated` / `weakly opposite` / `opposite`), calibrated against the cosine distribution from the query to the rest of the vocab on this specific checkpoint. Self-similarity sits at `identical` (z ≈ 20+); strong neighbours typically land at z ≥ 3 (`very similar`); near-mean cosines fall in `unrelated`. The reference `μ` and `σ` are printed alongside.

```bash
llm embeddings --help
```

### Subcommands at a glance

| Subcommand                 | What it does                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **`encode <text...>`**     | Tokenize input and emit the token-embedding row(s). `--per-token` keeps each row separate; default mean-pools. |
| **`decode <vector-spec>`** | Top-K nearest tokens to a vector. `<vector-spec>` is a word/phrase or `@path/to/vec.json`.                     |
| **`nearest <word>`**       | Convenience alias for `decode <word>`.                                                                         |
| **`similarity <a> <b>`**   | Cosine similarity in `[-1, 1]` between two inputs (each may be a word/phrase or `@file`).                      |
| **`expression "<expr>"`**  | Evaluate `+`, `-`, parens, unary `-` over embeddings and return the top-K nearest tokens to the result.        |

### Starter recipe (run these in order)

Pick a trained checkpoint and matching tokenizer:

```bash
MODEL=out/medium-longctx-push2
TOK=out/tok-8000.json
```

**1. Sanity-check the implementation.** A token compared against itself must score 1.0:

```bash
llm embeddings similarity --model $MODEL --tokenizer $TOK king king
```

**2. Verify the space encodes meaning.** Related pairs should land in `very similar` / `similar` (z ≥ 2), random controls in `unrelated` / `weakly similar`. Read the band, not the raw cosine:

```bash
llm embeddings similarity --model $MODEL --tokenizer $TOK he she
llm embeddings similarity --model $MODEL --tokenizer $TOK boy girl
llm embeddings similarity --model $MODEL --tokenizer $TOK he banana   # control
```

A strong run looks like `cosine: 0.158754  (very similar, z=3.2 vs reference μ=-0.003 σ=0.051)`.

**3. Look at neighbourhoods.** What does the model think `the` is similar to? On most corpora you’ll see other determiners / quantifiers (`another`, `such`, `local`, …) — that’s distributional similarity in action:

```bash
llm embeddings nearest --model $MODEL --tokenizer $TOK the --top-k 10
llm embeddings nearest --model $MODEL --tokenizer $TOK Sarah --top-k 10   # named entities cluster with names + pronouns
```

**4. The headline analogy.** This is the educational money shot — vector arithmetic over the embedding space:

```bash
llm embeddings expression --model $MODEL --tokenizer $TOK "king - man + woman" --top-k 10
```

Whether `queen` lands at rank 1 depends heavily on **corpus size and topic** (open-domain Wikipedia text reproduces the textbook result; small narrative-domain corpora typically surface female pronouns / names instead, which is _also_ a real signal: the gender axis is being learned). Both outcomes are pedagogically useful and the input-expansion trace tells you exactly what was tokenised.

**5. See the trick — toggle input exclusion.** By default we hide the literal input tokens from the result list, otherwise `king` itself dominates the top by trivial self-similarity. Compare:

```bash
llm embeddings expression --model $MODEL --tokenizer $TOK "king - man + woman" --top-k 10
llm embeddings expression --model $MODEL --tokenizer $TOK "king - man + woman" --top-k 10 --exclude-inputs=false
```

**6. Other relations worth probing.** Anything the model has seen enough of will show up:

```bash
llm embeddings expression --model $MODEL --tokenizer $TOK "father - he + she" --top-k 5     # should land near "mother" / female names
llm embeddings expression --model $MODEL --tokenizer $TOK "boys - boy + girl" --top-k 5     # plural axis (often noisier)
llm embeddings expression --model $MODEL --tokenizer $TOK "(king + queen) - man" --top-k 5  # parens compose
```

### About the auto-prefix space

Under the **`gpt2-v1`** pretokenizer, **`king`** and **`" king"`** (with a leading space) are _different_ tokens — and `" king"` is the variant that actually appears in real text (mid-sentence). So `llm embeddings` **prepends a single leading space to bare word inputs by default** (you’ll see `encoded as " king"` in the trace). Vector files (`@path.json`), inputs that already start with whitespace, and explicitly **quoted atoms inside `expression`** are left alone:

```bash
llm embeddings expression --model $MODEL --tokenizer $TOK 'king + "queen"' --top-k 5         # bare king → " king"; quoted "queen" stays literal
llm embeddings expression --model $MODEL --tokenizer $TOK "king - man + woman" --no-auto-space  # opt out globally
```

Use `--no-auto-space` (or quote individual atoms in expressions) to inspect the rare no-leading-space variants — useful for understanding BPE behaviour, but most of the time you want the default.

### A small expression-language reference

```text
expr   := term (('+' | '-') term)*
term   := factor
factor := atom | '(' expr ')' | '-' factor
atom   := identifier | '"' phrase '"'
```

- Identifiers match `[A-Za-z_][A-Za-z0-9_-]*`, so **`co-founder`** lexes as one atom; surround `-` with whitespace to make it an operator (**`man - woman`**).
- Quoted atoms can contain spaces (**`"new york"`**) and bypass auto-prefix.
- Multi-token atoms (e.g. words your tokenizer fragments into BPE pieces) are **mean-pooled** before they enter the expression — the trace shows when this happens (look for `[mean-pooled]`).

If results look like noise, check the trace first: a fragmented atom (`queen → " qu" + "een" [mean-pooled]`) tells you the corpus wasn’t big enough for `queen` to earn its own merge, and the analogy can’t name a token that doesn’t exist.

---

## Monorepo development

```bash
pnpm install
pnpm run build       # nx run-many -t build --all
pnpm run test       # nx run-many -t test --all
```

Package layout and OpenSpec/conventions for larger changes are described in **`docs/project.md`** and **`AGENTS.md`**.
