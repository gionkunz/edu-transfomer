// grad_in[i] = grad_out[i] * d/dx gelu(x) at x = inp[i] — same tanh GELU as elementwise-gelu.wgsl.
const GELU_TANH_COEF: f32 = 0.7978845608028654;
const GELU_CUBE_COEF: f32 = 0.044715;

fn gelu_grad_f32(x: f32) -> f32 {
  let u = GELU_TANH_COEF * (x + GELU_CUBE_COEF * x * x * x);
  let t = tanh(u);
  let dtanh = 1.0 - t * t;
  let dudx = GELU_TANH_COEF * (1.0 + 3.0 * GELU_CUBE_COEF * x * x);
  return 0.5 * (1.0 + t) + 0.5 * x * dtanh * dudx;
}

struct Params {
  element_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_in: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.element_count) {
    return;
  }
  let x = inp[i];
  grad_in[i] = grad_out[i] * gelu_grad_f32(x);
}
