## 1. Dependencies and entrypoint

- [x] 1.1 Add **`commander`** (or **`yargs`** if aligning with repo preference) and **`wolf-tui`** to **`apps/cli`** per **`docs/project.md`** allowlist; ensure versions match workspace policy.
- [x] 1.2 Add **`bin`** mapping for **`llm`** → compiled **`main.js`** (and ensure **`pnpm` / Nx build** produces a runnable file; patch ESM imports if the repo toolchain requires it).
- [x] 1.3 Replace placeholder **`main`** with programmatic **`createCliProgram()`** (or equivalent) callable from **`main`** for tests, plus **`main()`** invocation when run as **`llm`**.

## 2. Shared CLI infrastructure

- [x] 2.1 Implement **`ensureWebGpuDevice`** (or reuse **`packages/gpu`**) helpers with loud failures and clear messages when adapter/device unavailable.
- [x] 2.2 Centralize path resolution (**`--data`**, **`--out`**, **`--model`**, tokenizer paths) and user-facing **`Error`** handling (non-zero exit, no stack dumps in normal usage unless **`--verbose`**).

## 3. Tokenizer commands

- [x] 3.1 Implement **`llm tokenizer train`** using **`BPETokenizer`** (or documented train API): read corpus path(s), **`--vocab-size`**, **`--out`**, minimal progress suitable for stdout (progress bar optional; not full wolf-tui unless desired).
- [x] 3.2 Implement **`llm tokenizer encode`** and **`llm tokenizer decode`** loading saved tokenizer JSON; encode prints IDs; decode accepts comma-separated IDs per project examples.

## 4. Inspect command

- [x] 4.1 Implement **`llm inspect`** using **`packages/io`** to load **`model.json`** (and tensor manifest) and print config, parameter count estimate, and a short list of key tensor shapes to stdout.

## 5. Train command (wolf-tui)

- [x] 5.1 Add minimal **`Trainer` / loop hooks** if needed (**`onStep`** or similar on **`Trainer`**, **`Trainer.runEpoch`**, or a thin **`apps/cli`** loop calling **`trainStep`**) so the TUI can read loss, step index, LR, and tokens/sec without duplicating optimizer logic.
- [x] 5.2 Implement dataset glob + **`createLmBatch`** (or existing training helpers) wiring for **`llm train --data`** paths and **`--out`** checkpoint directory matching **`packages/io`** format.
- [x] 5.3 Build **wolf-tui** training dashboard: model label, step/epoch, scalar loss display (chart optional stretch), LR, ETA or elapsed; exit cleanly on **`q`** or Ctrl+C where wolf-tui allows.
- [x] 5.4 Verify **`llm train`** aborts loudly when GPU init fails.

## 6. Generate command (wolf-tui)

- [x] 6.1 Wire checkpoint tensor load (**`loadGptCheckpointTensors`** or **`io`** equivalents) into **`forwardGpt` / KV** path required by **`generateAutoregressive`**.
- [x] 6.2 Implement **wolf-tui** streaming view calling **`generateAutoregressive`** with prompt tokenization (`encode`/`decode` parity with training tokenizer path).
- [x] 6.3 Surface streaming metadata (tokens generated, approximate tokens/sec) in the status area; **`llm generate`** SHALL fail loudly without WebGPU.

## 7. Tests and manual verification

- [x] 7.1 Add Vitest coverage for **`inspect`** parsing and summaries using a fixture **`model.json`** (no GPU).
- [x] 7.2 Add tests for **`tokenizer`** CLI argv → handler delegation (mock filesystem or tiny temp dir) where practical.
- [x] 7.3 Document a short manual script in the PR or **`AGENTS`**-linked note: build → **`llm --help`** → one GPU smoke (`train` 1 step or `generate` 1 token) for local validation.
