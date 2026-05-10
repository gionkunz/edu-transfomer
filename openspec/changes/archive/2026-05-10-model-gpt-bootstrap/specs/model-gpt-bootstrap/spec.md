# model-gpt-bootstrap

## Requirements

### R1 — RMSNorm backward

The system SHALL provide GPU kernels **`core.rmsnorm_grad_input`** and **`core.rmsnorm_grad_gamma`** consistent with the forward **`core.rmsnorm`**, plus tensor helpers and an autograd **`rmsnorm`** wrapper.

### R2 — GPT forward

The model package SHALL expose **`forwardGpt(rt, model, tokenIds)`** returning rank-2 logits **`[seqLen, vocabSize]`** for rank-1 **`u32`** token indices, using **Pre-LN** blocks, **causal** multi-head attention (via masked logits), and a **GELU** FFN.

### R3 — Parameters

The model package SHALL provide **`collectGptParameters`** listing every **`Parameter`** owned by a **`GptModel`** (excluding constant **`headExpand`** leaves).
