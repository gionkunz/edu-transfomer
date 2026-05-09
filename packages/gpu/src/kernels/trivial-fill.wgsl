@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = arrayLength(&data);
  if (i >= n) {
    return;
  }
  data[i] = 3.14159265;
}
