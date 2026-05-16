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

| Command                               | Role                                                                                                                                                                                                                                |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`llm tokenizer train`**             | Fit **BPE** on text files matched by **`--data` globs**; write tokenizer JSON                                                                                                                                                       |
| **`llm tokenizer encode` / `decode`** | Tiny encode/decode utilities (stdout; no fullscreen TUI)                                                                                                                                                                            |
| **`llm train`**                       | Load corpus + tokenizer → **WebGPU** training → checkpoint dir (**wolf-tui** dashboard, including corpus **BPE encode** progress where applicable). Optional **`--dtype f32\|bf16\|fp16`** (default **`f32`**) for compute weights. |
| **`llm generate`**                    | Single prompt, streaming **wolf-tui**; **`--dtype`** and optional **`--allow-dtype-upcast`** for checkpoint vs runtime dtype.                                                                                                       |
| **`llm chat`**                        | Multi-turn **wolf-tui** (same sampling flags flavour as **`generate`)**; same dtype flags as **`generate`**.                                                                                                                        |
| **`llm inspect`**                     | Checkpoint / **`model.json`** summary on stdout                                                                                                                                                                                     |

For full flags, run **`llm train --help`**, **`llm tokenizer train --help`**, **`llm generate --help`**, **`llm chat --help`**, etc. Training and inference share the **`--dtype`** knob (`f32` / `bf16` / `fp16`); **`generate`** / **`chat`** also expose **`--allow-dtype-upcast`** when loading 16-bit checkpoints into an **`f32`** model—see **`docs/project.md`** (checkpoint + CLI sections).

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
