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

For full flags, run **`llm train --help`**, **`llm tokenizer train --help`**, **`llm generate --help`**, **`llm chat --help`**, etc. Training and inference share the **`--dtype`** knob (`f32` / `bf16`); **`generate`** / **`chat`** also expose **`--allow-dtype-upcast`** when loading **`bf16`** checkpoints into an **`f32`** model—see **`docs/project.md`** (checkpoint + CLI sections).

Built-in GPT shape **presets** for **`llm train --config`** (vocab comes from tokenizer only): **`tiny`**, **`small`**. Otherwise pass a **`GptConfig` JSON path**—see **`config/medium.json`** and **`config/medium-longctx.json`** in this repo (`vocabSize` in the file **must agree** with the tokenizer’s **`vocabSize`**).

---

## Example: train a tokenizer

Pick a **`--vocab-size`** that matches the model JSON you’ll use (e.g. **`8000`** for the sample **`config/medium*.json`** in this repo). **`--data`** is repeatable; each value is a glob.

```bash
llm tokenizer train --data 'corpus/real/*.txt' --vocab-size 8000 --out out/tok-8000.json
```

For several disjoint globs:

```bash
llm tokenizer train --data 'corpus/real/*.txt' --data 'notes/*.txt' --vocab-size 8000 --out out/tok-8000.json
```

> **BPE encode speed.** Tokenizers are trained under a mandatory **`word-and-punct-v1`** pre-tokenizer (split on whitespace / digits / Unicode letters / other). Encode walks each region independently and reaches quiescence in ~5 outer passes per region regardless of corpus size, dropping per-corpus encode from "hours" to "seconds" on 100 MB+ inputs. **Tokenizer JSON files saved before this change** no longer load (schema `version: 2` with an explicit `pretokenizer` label is required) — retrain to migrate. See `docs/project.md` → _BPE Tokenizer_ for the policy details.

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

## Monorepo development

```bash
pnpm install
pnpm run build       # nx run-many -t build --all
pnpm run test       # nx run-many -t test --all
```

Package layout and OpenSpec/conventions for larger changes are described in **`docs/project.md`** and **`AGENTS.md`**.
