# optimizer-training-utils

## Requirements

### R1 — Fused AdamW kernel

The GPU package SHALL register **`core.adamw_step`** executing a single fused AdamW step per element on contiguous **f32** buffers for parameters, gradients, first moment, and second moment, with host-supplied bias corrections and per-dispatch weight decay.

### R2 — Tensor surface

The tensor package SHALL expose **`adamWStepInPlace(rt, param, grad, m, v, hyper)`** for contiguous **f32** tensors of identical shape, delegating to **`dispatchAdamWStep`**.

### R3 — Gradient clipping

The autograd package SHALL provide **`clipGradNorm(rt, registry, maxNorm)`** computing the global L2 norm of all non-null Parameter gradients and scaling each gradient by **`maxNorm / norm`** when **`norm > maxNorm`**, using GPU tensor ops and an async readback for the norm.

### R4 — Training AdamW wrapper

The training package SHALL export an **`AdamW`** type that owns optimizer state tensors, supports **`setLr`**, applies **`adamWStepInPlace`** on **`step`**, and supports configurable per-parameter weight-decay scaling; **`dispose`** SHALL destroy moment buffers.
