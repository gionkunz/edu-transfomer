# embedding-cross-entropy

## Requirement: Embed gather/spatter on GPU

The system MUST provide WGSL kernels and tensor dispatch such that **`[vocab, embed]`** weight rows can be gathered by **`[tokens]`** `u32` indices and **`[tokens, embed]`** grads can accumulate into **`[vocab, embed]`** with correct handling of duplicate ids.

### Scenario: deterministic kernel IDs

- **WHEN** `registerCoreKernels` executes
- **THEN** kernels `core.embedding_gather` and `core.embedding_scatter_accum` MUST be registered uniquely

## Requirement: Softmax CE forward + logits grad

The system MUST provide numerically stabilized per-row cross-entropy from raw logits (**log-sum-exp** form) and logits gradients proportional to **`upstream_row * (p − one_hot(target))`** given softmax rows **`p`**.

### Scenario: composition with mean reduction

- **WHEN** autograd exposes `cross_entropy_mean_from_logits` composing row losses with `sum_flat` and scaling
- **THEN** logits gradients MUST match softmax CE analytic gradient with **`upstream_row_i = ∂L / ∂loss_i`**
