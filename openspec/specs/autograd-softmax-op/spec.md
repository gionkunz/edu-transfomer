# autograd-softmax-op

## Purpose

Specify the **pairwise** vertical slice for **row-wise softmax** on **rank-2** f32 tensors: a new `core.softmax_backward` WGSL kernel and dispatch in `packages/gpu`, **`tensor.softmax`** / **`tensor.softmaxBackward`** in `packages/tensor`, and **`softmax`** in `packages/autograd`, with CPU-reference forward checks and analytic or finite-difference backward checks. Forward `core.softmax` / `dispatchSoftmax` / `cpuSoftmax` are already present and remain behaviorally unchanged.

## Requirements

### Requirement: Softmax backward kernel and dispatch

The implementation SHALL register a WGSL compute pipeline under the stable identifier `core.softmax_backward`. For **row-major** `[rows, cols]` tensors, each row `i` SHALL compute the Jacobian–vector product of row-wise softmax with respect to logits using the forward output probabilities `y` and upstream gradients `grad_out`:

\[
\text{grad_in}_{ij} = y_{ij}\left(\text{grad_out}_{ij} - \sum_k y_{ik}\,\text{grad_out}\_{ik}\right)
\]

The implementation SHALL expose `dispatchSoftmaxBackward(ctx, registry, gradOut, y, gradIn, rows, cols)` with the documented bind-group layout and the **same row-dispatch pattern** as `dispatchSoftmax` (one logical row per invocation index, uniform `(rows, cols)` matching existing softmax kernels).

#### Scenario: GPU output matches CPU reference

- **WHEN** tests run `dispatchSoftmaxBackward` against `cpuSoftmaxBackward` for small `[rows, cols]` with WebGPU available
- **THEN** readback of `grad_in` matches the CPU reference within floating-point tolerance

---

### Requirement: Tensor ops `softmax` and `softmaxBackward`

The implementation SHALL expose `softmax(rt, a)` for **rank-2** tensors only, with the **same shape and contiguity preconditions** as other rank-2 ops in `packages/tensor`, allocating a fresh output buffer and dispatching **`core.softmax`** via the existing softmax dispatch helper.

The implementation SHALL expose `softmaxBackward(rt, gradOutput, softmaxOutput)` requiring `gradOutput` and `softmaxOutput` to have **equal shape** to each other and to the rank-2 layout expected by the backward kernel, where **`softmaxOutput` is the forward softmax probability tensor `y`**. The function SHALL allocate a fresh buffer and dispatch **`core.softmax_backward`**.

#### Scenario: Forward matches CPU reference

- **WHEN** `softmax` is called on tensor data filled from a host array and results are read back with WebGPU available
- **THEN** output equals `cpuSoftmax` on that data within tolerance

#### Scenario: Backward matches CPU reference

- **WHEN** `softmaxBackward` is called with random-like `gradOutput` and `softmaxOutput` produced from the same row shape
- **THEN** output equals `cpuSoftmaxBackward` within tolerance

---

### Requirement: Autograd wrapper `softmax`

The implementation SHALL expose `softmax(rt, x: Variable): Variable` whose forward tensor equals `tensor.softmax(rt, x.tensor)`. When `x.requiresGrad === true`, the recorded `Operation` SHALL have `name: 'softmax'`, `inputs: [x]`, and `backward(rt, gradOutput)` returning `[tensor.softmaxBackward(rt, gradOutput, y)]` where **`y` is the tensor returned by the forward `tensor.softmax` call** (saved in the closure). When `x.requiresGrad === false`, the wrapper MAY return a leaf variable without recording an `Operation`.

#### Scenario: Backward uses forward probability tensor

- **WHEN** `backward` is invoked with a non-zero `gradOutput`
- **THEN** the input gradient equals `tensor.softmaxBackward(rt, gradOutput, y)` within tolerance

#### Scenario: Finite-difference or analytic agreement

- **WHEN** a test compares analytic gradients from `backward` to central-difference numerical gradients for a small rank-2 `softmax` graph
- **THEN** the error is within the test’s documented tolerance

---

### Requirement: Explicit exclusions

The implementation SHALL NOT depend on softmax backward that **computes derivatives on the CPU and uploads** them as the primary training path for this op. The implementation SHALL NOT add **RMSNorm**, **cross-entropy**, **attention masking**, **sliding-window softmax**, **non-f32** dtypes, or **N-D / rank ≠ 2** softmax beyond the `[rows, cols]` kernel contract in this change.
