## Why

Sequential plan step **11** (`docs/sequential-change-proposals.md`) needs a minimal **training loop** in `packages/training`: LM batching helpers, cosine LR with warmup, a **`Trainer`** that composes forward + loss + backward + optional grad clip + **`AdamW`**, and **GPT checkpoint save/load** paired with **`packages/io`**. Exit criterion: a tiny run shows decreasing loss and checkpoints round-trip tensor payloads.

## What Changes

- **`packages/training`**: **`getCosineLr`** / **`CosineLrSchedule`**, **`createLmBatch`** (next-token shift), **`Trainer`** (**`registerModelParameters`**), **`saveGptCheckpoint`** / **`loadGptCheckpointTensors`**; Vitests for scheduler (pure) and trainer (GPU: overfit + checkpoint key count).
- **`packages/model`**: **`gptParameterTensorNames`** aligned with **`collectGptParameters`** order for stable checkpoint keys; exported from **`index`**.

## Explicitly Out of Scope

- Full dataset loaders, DDP, logging/TUI, optimizer state in checkpoints, inference KV cache, CLI **`train`** command.

## Capabilities

### New

- **`training-loop-bootstrap`**: LM batch helper, cosine LR schedule, trainer step, GPT checkpoint I/O wiring, public exports from **`training`**.

### Modified

- **`model-gpt-bootstrap`**: stable parameter naming surface for checkpoints (**implementation detail of step 10 package**).

## Impact

- Primary: `packages/training`, `packages/model` (small export + naming helper).
- Consumers: future CLI and larger training scripts depend on **`Trainer`** + checkpoint helpers without new ML dependencies.
