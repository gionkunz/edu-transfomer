// Fused AdamW step on 1D parallel slices: updates θ, m, v in place; reads g.
struct AdamUniforms {
  n_pad: vec4<u32>,
  hp0: vec4<f32>, // lr, beta1, beta2, eps
  hp1: vec4<f32>, // weight_decay, bias_corr1, bias_corr2, unused
}

@group(0) @binding(0) var<uniform> ub: AdamUniforms;
@group(0) @binding(1) var<storage, read_write> theta: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = ub.n_pad.x;
  if (i >= n) {
    return;
  }
  let lr = ub.hp0.x;
  let beta1 = ub.hp0.y;
  let beta2 = ub.hp0.z;
  let eps = ub.hp0.w;
  let wd = ub.hp1.x;
  let bc1 = ub.hp1.y;
  let bc2 = ub.hp1.z;

  let g = grad[i];
  let m_new = beta1 * m[i] + (1.0 - beta1) * g;
  let v_new = beta2 * v[i] + (1.0 - beta2) * g * g;
  let m_hat = m_new / bc1;
  let v_hat = v_new / bc2;
  let denom = sqrt(v_hat) + eps;
  theta[i] = theta[i] - lr * m_hat / denom - lr * wd * theta[i];
  m[i] = m_new;
  v[i] = v_new;
}
