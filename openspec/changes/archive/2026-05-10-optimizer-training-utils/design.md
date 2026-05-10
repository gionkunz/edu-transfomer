## AdamW

- **WGSL** implements one thread per element updating **θ**, **m**, **v**; **grad** is read-only. Uniform block packs `n`, `lr`, `β₁`, `β₂`, `ε`, **`weight_decay`** (host applies per-parameter multiplier before dispatch), and bias corrections **`1−β₁ᵗ`**, **`1−β₂ᵗ`** computed on CPU.
- **Formula** matches `docs/project.md`: `θ ← θ − lr·m̂/(√v̂+ε) − lr·λ·θ` with **m̂**, **v̂** using stored **m**, **v** after EMA update.
- **Tensor** exposes **`adamWStepInPlace`** so **`training`** does not import **`gpu`** at runtime (layering: training → tensor → gpu via tensor’s existing edge).

## Gradient clipping

- **Global norm²** = sum over parameters of **`sumFlat(mul(g,g))`** with **`g`** reshaped to rank-1 when needed; fused **GPU** reductions, single readback for **`sqrt`**.
- **Scale** uses existing **`scale`**; replaces **`p.grad`** tensors and **`destroy`**s previous grad buffers to avoid leaks.

## AdamW class

- **Lazy** **m**/**v** **`createZeros`** on first **`step`** for each **`Parameter`**.
- **`weightDecayScale`**: `(param) => number` default **`1`**; use **`0`** for biases / norms / embeddings per project guidance.
- **`step`** requires **`p.grad`** set; after update **`destroy(grad)`**, **`p.grad = null`** (caller runs **`zeroGrad`** at loop start per project snippet).
