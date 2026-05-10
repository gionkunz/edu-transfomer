## Why

Sequential step **10** (`docs/sequential-change-proposals.md`) needs a **decoder-only GPT** forward path (Pre-LN, causal attention, FFN) wired through **autograd** so training can optimize real blocks. RMSNorm forward existed on GPU but had **no tensor/autograd** path or backward kernels.

## What Changes

- **`packages/gpu`**: **`core.rmsnorm_grad_input`**, **`core.rmsnorm_grad_gamma`** WGSL + dispatch + **`cpuRmsNormGrad*`**; GPU tests vs CPU.
- **`packages/tensor`**: **`rmsnorm`**, **`rmsnormGradInput`**, **`rmsnormGradGamma`**.
- **`packages/autograd`**: **`rmsnorm`** op.
- **`packages/model`**: **`GptConfig`**, **`createGptModel`**, **`forwardGpt`**, **`collectGptParameters`**, **`buildCausalBiasTensor`**; **multi-head attention** via **per-head Q/K/V projections** (`[d_model, d_k]`) and **fixed sparse `headExpand` matrices** (`[d_k, d_model]`) to accumulate head outputs into **`[T, d_model]`** without a dedicated permute kernel (documented tradeoff); **Pre-LN** + **GELU** FFN + **weight + pos embeddings**; Vitest forward + backward smoke test; depends on **`tensor`**; **`gpu`** devDependency for tests.

## Explicitly Out of Scope

- Weight tying, KV-cache inference path, fused attention, RoPE, dropout, LayerNorm, bias in linear layers, checkpoint load in model, larger-than-GPU-tested sequence hardening.

## Capabilities

### New

- **`model-gpt-bootstrap`**: Minimal trainable GPT assembly and RMSNorm backward in the stack.

### Modified

- **`reduction-norm-kernels`** (inventory): `rmsnorm_grad_*` IDs.

## Impact

- Enables **`packages/training`** to run a toy LM loop next; **`inference`** can reuse **`forwardGpt`** later (KV cache still separate).
