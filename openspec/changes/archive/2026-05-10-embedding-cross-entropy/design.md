## Decisions

- **Embedding backward** uses WGSL bitwise **atomic float CAS** accumulation so duplicate ids in a minibatch accumulate correctly (`core.embedding_scatter_accum`). Callers allocate a zeroed dense **`[vocab, embed]`** buffer per backward edge.
- **Cross-entropy logits grad** consumes **row-softmax probabilities** `p` (`core.cross_entropy_logits_grad`) with a **per-row upstream** buffer so autograd compositions (e.g. mean over examples) propagate without host scalar readback.
- **`core.broadcast_fill_scalar`** fills an `n`‑vector from the first slot of a `[1]` buffer so **`sumFlat`** backward can reuse existing tensor buffers only.

## Compatibility

Additive public surface; **`tensor.reshape`** now asserts **`dtype: 'f32'`** (pure metadata reshapes on `u32` indices are unsupported until needed).
