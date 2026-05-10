## RMSNorm backward

- Matches standard derivative for `y = x * inv_rms * gamma` with `inv_rms = 1/sqrt(mean(x^2)+eps)` per row.
- **Grad-γ**: column reduction over rows (recomputes `inv` per row).

## Multi-head layout

- **Per-head** `Wq/Wk/Wv` are **`[d_model, d_k]`** (compact vs single stacked `[d_model, d_model]`).
- **Head merge**: `mixed_h [T, d_k] @ E_h [d_k, d_model]` where **`E_h`** is zeros except an identity block in columns **`[h*d_k, (h+1)*d_k)`**; **`E_h`** are **non-trainable** `Variable` leaves.

## Causal mask

- Dense **`[T, T]`** bias tensor with **`0`** on/off-diagonal lower triangle and **`-1e4`** strictly upper; added to attention **logits** before **softmax**.
