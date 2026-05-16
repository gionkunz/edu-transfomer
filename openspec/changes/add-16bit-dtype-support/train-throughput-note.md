# Train §7.5 — VRAM / throughput (best-effort)

OpenSpec task **9.5** points here for the “evidence the success criterion requires”: fill the table below when you have hardware measurements (same note as **7.5**).

OpenSpec task **7.5** asks for a measurable **VRAM** and **tokens/sec** comparison for the **`small`** preset in **f32** vs **bf16** on a reference adapter.

This note is intentionally **not** automated in CI: reporting GPU memory fairly depends on the WebGPU/Dawn runtime, driver, and OS tools (`nvidia-smi`, Task Manager, etc.).

## How to reproduce

1. Build the CLI (`npm run build` or Nx targets under `AGENTS.md`).
2. Run `llm train` twice with the same **corpus**, **`--steps`**, **`--seq-len`**, and optimizer flags, changing only **`--dtype f32`** vs **`--dtype bf16`**.
3. Note **`tokens/sec`** from the training dashboard / stderr timing if logged, or derive from wall time and **`steps × seq-len`**.
4. Capture **peak GPU memory** with your platform tool while the run is active.

## Result placeholder

| Adapter / OS | preset | dtype | peak VRAM (tool) | approx tokens/sec |
| ------------ | ------ | ----- | ---------------- | ----------------- |
| _fill in_    | small  | f32   |                  |                   |
| _fill in_    | small  | bf16  |                  |                   |

Archive this file (or paste the filled table into the change archive) when closing the OpenSpec change.
