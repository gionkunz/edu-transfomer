// out[i] = a[i] * scale (1D f32 buffers). scale comes from uniform next to element_count.
struct ScaleParams {
  element_count: u32,
  scale: f32,
  _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> scale_params: ScaleParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= scale_params.element_count) {
    return;
  }
  out[i] = a[i] * scale_params.scale;
}
