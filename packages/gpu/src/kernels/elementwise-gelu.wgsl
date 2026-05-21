// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) — matches cpu-reference.ts.
// For |x| > GELU_LINEAR_THRESHOLD use the linear tail (GELU≈x or 0) instead of tanh(u)
// with large u — some drivers (D3D12 / Blackwell) return NaN from tanh at large |u|.
const GELU_TANH_COEF: f32 = 0.7978845608028654;
const GELU_CUBE_COEF: f32 = 0.044715;
const GELU_LINEAR_THRESHOLD: f32 = 8.0;

fn gelu_f32(x: f32) -> f32 {
  if (x > GELU_LINEAR_THRESHOLD) {
    return x;
  }
  if (x < -GELU_LINEAR_THRESHOLD) {
    return 0.0;
  }
  let t = tanh(GELU_TANH_COEF * (x + GELU_CUBE_COEF * x * x * x));
  return 0.5 * x * (1.0 + t);
}

struct Params {
  element_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.element_count) {
    return;
  }
  out[i] = gelu_f32(inp[i]);
}
