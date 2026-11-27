// WGSL compute shaders that run the physics neural solver (the 3D U-Net of physics_neural_solver/model.py) on the GPU.
//
// Memory layout = PyTorch's: a tensor (C, X, Y, Z) is stored channel by channel, x slowest and z fastest:
//     index = ((c * X + x) * Y + y) * Z + z
// so the weights from weights.bin can be used exactly as PyTorch stores them.
// Every kernel handles one output number per thread; large dispatches use 2D workgroup grids.

const INDEX = /* wgsl */ `
fn threadIndex(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
  return gid.x + gid.y * groups.x * 64u;                       // 1D index from a 2D grid of 64-thread workgroups
}`;

// ---------------------------------------------------------------------------------------------------------------
// 3D convolution, kernel size 3 (padding 1, zeros outside) or 1:   out[o] = bias[o] + sum_i sum_k w[o, i, k] * in[i, x + k]
// ---------------------------------------------------------------------------------------------------------------
export const CONVOLUTION = /* wgsl */ `
struct Params { cin: u32, cout: u32, nx: u32, ny: u32, nz: u32, k: u32, weight_offset: u32, bias_offset: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
${INDEX}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let cells = P.nx * P.ny * P.nz;                              // cells of this level
  let index = threadIndex(gid, groups);                        // which output number
  if (index >= P.cout * cells) { return; }
  let o = index / cells;                                       // output channel
  let cell = index % cells;                                    // output cell
  let x = i32(cell / (P.ny * P.nz));
  let y = i32((cell / P.nz) % P.ny);
  let z = i32(cell % P.nz);
  let r = i32(P.k / 2u);                                       // 1 for a 3x3x3 kernel, 0 for 1x1x1
  let k3 = P.k * P.k * P.k;                                    // weights per (output, input) pair
  var sum = weights[P.bias_offset + o];                        // start with the bias
  for (var i = 0u; i < P.cin; i = i + 1u) {                    // every input channel
    let in_base = i * cells;
    let w_base = P.weight_offset + (o * P.cin + i) * k3;
    for (var dx = 0; dx < i32(P.k); dx = dx + 1) {
      let xx = x + dx - r;
      if (xx < 0 || xx >= i32(P.nx)) { continue; }             // zero padding
      for (var dy = 0; dy < i32(P.k); dy = dy + 1) {
        let yy = y + dy - r;
        if (yy < 0 || yy >= i32(P.ny)) { continue; }
        for (var dz = 0; dz < i32(P.k); dz = dz + 1) {
          let zz = z + dz - r;
          if (zz < 0 || zz >= i32(P.nz)) { continue; }
          let input_index = in_base + (u32(xx) * P.ny + u32(yy)) * P.nz + u32(zz);
          let weight_index = w_base + (u32(dx) * P.k + u32(dy)) * P.k + u32(dz);
          sum = sum + input[input_index] * weights[weight_index];
        }
      }
    }
  }
  output[index] = sum;
}`;

// ---------------------------------------------------------------------------------------------------------------
// GroupNorm, step 1: mean and 1/std of every group (one workgroup per group; groups are runs of whole channels)
// ---------------------------------------------------------------------------------------------------------------
export const GROUP_NORM_STATISTICS = /* wgsl */ `
struct Params { channels: u32, groups: u32, cells: u32, gamma_offset: u32, beta_offset: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> statistics: array<f32>;
var<workgroup> partial: array<f32, 256>;
fn reduce(local: u32) {                                         // sum partial[0..255] into partial[0]
  var stride = 128u;
  loop {
    workgroupBarrier();
    if (local < stride) { partial[local] = partial[local] + partial[local + stride]; }
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  workgroupBarrier();
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wg.x;                                                // which group
  let count = (P.channels / P.groups) * P.cells;               // numbers in this group
  let base = g * count;                                        // its channels are next to each other in memory
  var s = 0.0;
  for (var i = lid.x; i < count; i = i + 256u) { s = s + data[base + i]; }
  partial[lid.x] = s;
  reduce(lid.x);
  let mean = partial[0] / f32(count);                          // group mean
  workgroupBarrier();
  var v = 0.0;
  for (var i = lid.x; i < count; i = i + 256u) { let d = data[base + i] - mean; v = v + d * d; }
  partial[lid.x] = v;
  reduce(lid.x);
  if (lid.x == 0u) {
    statistics[2u * g] = mean;                                 // mean
    statistics[2u * g + 1u] = inverseSqrt(partial[0] / f32(count) + 1e-5);   // 1 / sqrt(variance + eps), as PyTorch
  }
}`;

// GroupNorm, step 2: normalise, scale and shift (gamma, beta), then SiLU:  y = x * sigmoid(x)
export const GROUP_NORM_APPLY = /* wgsl */ `
struct Params { channels: u32, groups: u32, cells: u32, gamma_offset: u32, beta_offset: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var<storage, read> statistics: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
${INDEX}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let index = threadIndex(gid, groups);
  if (index >= P.channels * P.cells) { return; }
  let c = index / P.cells;                                     // channel
  let g = c / (P.channels / P.groups);                         // its group
  let normalised = (data[index] - statistics[2u * g]) * statistics[2u * g + 1u];
  let y = normalised * weights[P.gamma_offset + c] + weights[P.beta_offset + c];
  data[index] = y / (1.0 + exp(-y));                           // SiLU
}`;

// ---------------------------------------------------------------------------------------------------------------
// Max pooling 2x2x2 (halves the grid)
// ---------------------------------------------------------------------------------------------------------------
export const MAX_POOL = /* wgsl */ `
struct Params { channels: u32, nx: u32, ny: u32, nz: u32 };    // OUTPUT size
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
${INDEX}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let cells = P.nx * P.ny * P.nz;
  let index = threadIndex(gid, groups);
  if (index >= P.channels * cells) { return; }
  let c = index / cells;
  let cell = index % cells;
  let x = cell / (P.ny * P.nz);
  let y = (cell / P.nz) % P.ny;
  let z = cell % P.nz;
  let ix = 2u * P.nx; let iy = 2u * P.ny; let iz = 2u * P.nz; // input size
  var best = -3.4e38;
  for (var a = 0u; a < 2u; a = a + 1u) {
    for (var b = 0u; b < 2u; b = b + 1u) {
      for (var d = 0u; d < 2u; d = d + 1u) {
        let value = input[((c * ix + 2u * x + a) * iy + 2u * y + b) * iz + 2u * z + d];
        best = max(best, value);
      }
    }
  }
  output[index] = best;
}`;

// ---------------------------------------------------------------------------------------------------------------
// Transposed convolution, kernel 2, stride 2 (doubles the grid). PyTorch weight shape: (cin, cout, 2, 2, 2)
//   out[o, X] = bias[o] + sum_i in[i, X / 2] * w[i, o, X % 2]
// ---------------------------------------------------------------------------------------------------------------
export const UPSAMPLE = /* wgsl */ `
struct Params { cin: u32, cout: u32, nx: u32, ny: u32, nz: u32, weight_offset: u32, bias_offset: u32, p0: u32 };  // OUTPUT size
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
${INDEX}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let cells = P.nx * P.ny * P.nz;
  let index = threadIndex(gid, groups);
  if (index >= P.cout * cells) { return; }
  let o = index / cells;
  let cell = index % cells;
  let x = cell / (P.ny * P.nz);
  let y = (cell / P.nz) % P.ny;
  let z = cell % P.nz;
  let hx = P.nx / 2u; let hy = P.ny / 2u; let hz = P.nz / 2u; // input size
  let kernel = ((x % 2u) * 2u + (y % 2u)) * 2u + (z % 2u);     // which of the 8 kernel weights
  let input_cell = ((x / 2u) * hy + y / 2u) * hz + z / 2u;
  var sum = weights[P.bias_offset + o];
  for (var i = 0u; i < P.cin; i = i + 1u) {
    sum = sum + input[i * hx * hy * hz + input_cell] * weights[P.weight_offset + (i * P.cout + o) * 8u + kernel];
  }
  output[index] = sum;
}`;

// Copy `count` numbers into a buffer at an offset (the skip connection goes after the upsampled channels)
export const COPY = /* wgsl */ `
struct Params { count: u32, destination_offset: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> source: array<f32>;
@group(0) @binding(2) var<storage, read_write> destination: array<f32>;
${INDEX}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let index = threadIndex(gid, groups);
  if (index >= P.count) { return; }
  destination[P.destination_offset + index] = source[index];
}`;

// ---------------------------------------------------------------------------------------------------------------
// Network input (15 channels, scenes.py + model.network_input), built on the padded grid (edge values copied)
// ---------------------------------------------------------------------------------------------------------------
export const ASSEMBLE_INPUT = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, nz: u32, stone_active: u32,                // grid (with walls)
  px: u32, py: u32, pz: u32, p0: u32,                          // padded grid
  h: f32, half_x: f32, half_y: f32, half_z: f32,               // cell size and half the box size (dimensionless)
  gravity: vec3<f32>, tilt_rate: f32,
  stone_centre: vec3<f32>, tilt_acceleration: f32,
  stone_velocity: vec3<f32>, stone_radius: f32,
  stone_spin: vec3<f32>, p1: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> phi: array<f32>;
@group(0) @binding(2) var<storage, read> potential: array<f32>;
@group(0) @binding(3) var<storage, read> pressure: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
${INDEX}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let padded = P.px * P.py * P.pz;
  let index = threadIndex(gid, groups);
  if (index >= padded) { return; }
  let x = min(index / (P.py * P.pz), P.nx - 1u);               // padding copies the edge cell
  let y = min((index / P.pz) % P.py, P.ny - 1u);
  let z = min(index % P.pz, P.nz - 1u);
  let cells = P.nx * P.ny * P.nz;
  let n = (x * P.ny + y) * P.nz + z;                           // the grid cell
  let position = (vec3<f32>(f32(x), f32(y), f32(z)) - 0.5) * P.h - vec3<f32>(P.half_x, P.half_y, P.half_z);
  let wall = x == 0u || y == 0u || z == 0u || x == P.nx - 1u || y == P.ny - 1u || z == P.nz - 1u;
  let offset = position - P.stone_centre;
  let in_stone = P.stone_active == 1u && length(offset) < P.stone_radius;
  var stone_velocity = vec3<f32>(0.0);
  if (in_stone) { stone_velocity = P.stone_velocity + cross(P.stone_spin, offset); }   // V + omega x (x - x_s)
  output[0u * padded + index] = phi[n];
  output[1u * padded + index] = potential[0u * cells + n];
  output[2u * padded + index] = potential[1u * cells + n];
  output[3u * padded + index] = potential[2u * cells + n];
  output[4u * padded + index] = pressure[n] - max(phi[n], 0.0);   // the moving part of the pressure (minus the still-water part)
  output[5u * padded + index] = select(0.0, 1.0, wall);
  output[6u * padded + index] = select(0.0, 1.0, in_stone);
  output[7u * padded + index] = stone_velocity.x;
  output[8u * padded + index] = stone_velocity.y;
  output[9u * padded + index] = stone_velocity.z;
  output[10u * padded + index] = P.gravity.x;
  output[11u * padded + index] = P.gravity.y;
  output[12u * padded + index] = P.gravity.z;
  output[13u * padded + index] = P.tilt_rate;
  output[14u * padded + index] = P.tilt_acceleration;
}`;

// ---------------------------------------------------------------------------------------------------------------
// Mean of the change of A over the (unpadded) grid: A is only defined up to a constant (one workgroup per component)
// ---------------------------------------------------------------------------------------------------------------
export const POTENTIAL_CHANGE_MEAN = /* wgsl */ `
struct Params { nx: u32, ny: u32, nz: u32, p0: u32, px: u32, py: u32, pz: u32, p1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> change: array<f32>;
@group(0) @binding(2) var<storage, read_write> mean: array<f32>;
var<workgroup> partial: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let c = wg.x;                                                // component of A (output channel 0, 1, 2)
  let cells = P.nx * P.ny * P.nz;
  let padded = P.px * P.py * P.pz;
  var s = 0.0;
  for (var i = lid.x; i < cells; i = i + 256u) {
    let x = i / (P.ny * P.nz); let y = (i / P.nz) % P.ny; let z = i % P.nz;
    s = s + change[c * padded + (x * P.py + y) * P.pz + z];   // read inside the padded output
  }
  partial[lid.x] = s;
  var stride = 128u;
  loop {
    workgroupBarrier();
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  if (lid.x == 0u) { mean[c] = partial[0] / f32(cells); }
}`;

// ---------------------------------------------------------------------------------------------------------------
// Apply the network output (model.PhysicsNeuralSolver.forward):
//   A = (A + dA - mean(dA)) x wall mask    (eq. 1 + eq. 10: A's in-wall components are 0 on the six wall planes)
//   p = p - max(phi, 0) + dp               (the new MOVING pressure; the still-water part of the new surface is
//                                           added after the surface has moved: MASK_PRESSURE)
// ---------------------------------------------------------------------------------------------------------------
export const APPLY_CHANGE = /* wgsl */ `
struct Params { nx: u32, ny: u32, nz: u32, p0: u32, px: u32, py: u32, pz: u32, p1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> change: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read_write> potential: array<f32>;
@group(0) @binding(4) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(5) var<storage, read> phi: array<f32>;
${INDEX}
fn inWallPlane(c: u32, x: u32, y: u32, z: u32) -> bool {       // eq. 10: this component of A lies in a wall plane
  let xw = x == 0u || x >= P.nx - 2u;                          // the x-wall planes (index 0, n-2) + the plane outside (n-1)
  let yw = y == 0u || y >= P.ny - 2u;                          // floor and lid
  let zw = z == 0u || z >= P.nz - 2u;                          // front and back
  if (c == 0u) { return yw || zw; }                            // A_x lies in floor/lid and front/back planes
  if (c == 1u) { return xw || zw; }                            // A_y in the x-walls and front/back
  return xw || yw;                                             // A_z in the x-walls and floor/lid
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let cells = P.nx * P.ny * P.nz;
  let index = threadIndex(gid, groups);
  if (index >= cells) { return; }
  let x = index / (P.ny * P.nz); let y = (index / P.nz) % P.ny; let z = index % P.nz;
  let padded = P.px * P.py * P.pz;
  let p_index = (x * P.py + y) * P.pz + z;                     // the same cell in the padded output
  for (var c = 0u; c < 3u; c = c + 1u) {
    let a = potential[c * cells + index] + change[c * padded + p_index] - mean[c];
    potential[c * cells + index] = select(a, 0.0, inWallPlane(c, x, y, z));
  }
  pressure[index] = pressure[index] - max(phi[index], 0.0) + change[3u * padded + p_index];   // moving part
}`;

// Shared by the strict-rule kernels below
const GRID = /* wgsl */ `
struct Grid { nx: u32, ny: u32, nz: u32, axis: u32, h: f32, dt: f32, p0: f32, p1: f32 };`;

// ---------------------------------------------------------------------------------------------------------------
// Eq. 5 built in: the surface is CARRIED by the water (semi-Lagrangian, RK2), u = (u_now + u_next) / 2.
// Index units: cell (i, j, k) has its centre at (i, j, k); u_x lives at i + 1/2, u_y at j + 1/2, u_z at k + 1/2.
// (physical_equations.carry_surface / velocity_at / sample_trilinear)
// ---------------------------------------------------------------------------------------------------------------
export const CARRY_SURFACE = /* wgsl */ `
${GRID}
@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read> phi: array<f32>;
@group(0) @binding(2) var<storage, read> u_now: array<f32>;
@group(0) @binding(3) var<storage, read> u_next: array<f32>;
@group(0) @binding(4) var<storage, read_write> phi_out: array<f32>;
${INDEX}
fn trilinear(which: u32, c: u32, g_in: vec3<f32>) -> f32 {     // which: 0 = phi, 1 = u_now, 2 = u_next; c = component
  let n = vec3<f32>(f32(G.nx), f32(G.ny), f32(G.nz));
  let g = clamp(g_in, vec3<f32>(0.0), n - 1.0);                // stay inside the grid
  let i0 = min(floor(g), n - 2.0);                             // lower corner of the surrounding cube
  let f = g - i0;                                              // position inside the cube
  let base = c * G.nx * G.ny * G.nz;
  var total = 0.0;
  for (var dx = 0u; dx < 2u; dx = dx + 1u) {
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
      for (var dz = 0u; dz < 2u; dz = dz + 1u) {
        let i = u32(i0.x) + dx; let j = u32(i0.y) + dy; let k = u32(i0.z) + dz;
        let n_index = base + (i * G.ny + j) * G.nz + k;
        var value = 0.0;
        if (which == 0u) { value = phi[n_index]; } else if (which == 1u) { value = u_now[n_index]; } else { value = u_next[n_index]; }
        let w = select(1.0 - f.x, f.x, dx == 1u) * select(1.0 - f.y, f.y, dy == 1u) * select(1.0 - f.z, f.z, dz == 1u);
        total = total + w * value;
      }
    }
  }
  return total;
}
fn velocityAt(g: vec3<f32>) -> vec3<f32> {                     // u_mid at index position g
  let gx = g - vec3<f32>(0.5, 0.0, 0.0); let gy = g - vec3<f32>(0.0, 0.5, 0.0); let gz = g - vec3<f32>(0.0, 0.0, 0.5);
  return 0.5 * vec3<f32>(trilinear(1u, 0u, gx) + trilinear(2u, 0u, gx),
                         trilinear(1u, 1u, gy) + trilinear(2u, 1u, gy),
                         trilinear(1u, 2u, gz) + trilinear(2u, 2u, gz));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let cells = G.nx * G.ny * G.nz;
  let index = threadIndex(gid, groups);
  if (index >= cells) { return; }
  let g = vec3<f32>(f32(index / (G.ny * G.nz)), f32((index / G.nz) % G.ny), f32(index % G.nz));   // this cell centre
  let cells_per_step = G.dt / G.h;                             // velocity -> cells per time step
  let mid = g - 0.5 * cells_per_step * velocityAt(g);          // half a step back (RK2 midpoint)
  let back = g - cells_per_step * velocityAt(mid);             // where this water was one step ago
  phi_out[index] = trilinear(0u, 0u, back);                    // phi from there
}`;

// Wall cells copy the nearest inside cell (one axis per dispatch, x then y then z, like copy_walls_from_inside)
export const COPY_WALLS_FROM_INSIDE = /* wgsl */ `
${GRID}
@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read_write> field: array<f32>;
${INDEX}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let index = threadIndex(gid, groups);
  if (index >= G.nx * G.ny * G.nz) { return; }
  let cell = vec3<u32>(index / (G.ny * G.nz), (index / G.nz) % G.ny, index % G.nz);
  let n = vec3<u32>(G.nx, G.ny, G.nz);
  let a = cell[G.axis];                                        // position along this axis
  if (a != 0u && a != n[G.axis] - 1u) { return; }              // only the two wall layers of this axis
  var inside = cell;                                           // the nearest inside cell
  if (a == 0u) { inside[G.axis] = 1u; } else { inside[G.axis] = n[G.axis] - 2u; }
  field[index] = field[(inside.x * G.ny + inside.y) * G.nz + inside.z];
}`;

// ---------------------------------------------------------------------------------------------------------------
// Eq. 7 built in: water volume and the Newton shift that restores it (one workgroup sums the whole grid).
// stats[0] = volume, stats[1] = d(volume)/d(shift), stats[2] = shift.  (physical_equations.restore_volume)
// ---------------------------------------------------------------------------------------------------------------
export const VOLUME_SUM = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, nz: u32, stone_active: u32, px: u32, py: u32, pz: u32, p0: u32,
  h: f32, half_x: f32, half_y: f32, half_z: f32,
  gravity: vec3<f32>, tilt_rate: f32, stone_centre: vec3<f32>, tilt_acceleration: f32,
  stone_velocity: vec3<f32>, stone_radius: f32, stone_spin: vec3<f32>, p1: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> phi: array<f32>;
@group(0) @binding(2) var<storage, read> target_volume: array<f32>;
@group(0) @binding(3) var<storage, read_write> stats: array<f32>;
var<workgroup> volume_part: array<f32, 256>;
var<workgroup> slope_part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let cells = P.nx * P.ny * P.nz;
  var v = 0.0; var s = 0.0;
  for (var index = lid.x; index < cells; index = index + 256u) {
    let x = index / (P.ny * P.nz); let y = (index / P.nz) % P.ny; let z = index % P.nz;
    if (x == 0u || y == 0u || z == 0u || x == P.nx - 1u || y == P.ny - 1u || z == P.nz - 1u) { continue; }   // walls
    let position = (vec3<f32>(f32(x), f32(y), f32(z)) - 0.5) * P.h - vec3<f32>(P.half_x, P.half_y, P.half_z);
    if (P.stone_active == 1u && length(position - P.stone_centre) < P.stone_radius) { continue; }      // inside the stone
    let H = 0.5 * (1.0 + tanh(phi[index] / (1.5 * P.h)));      // smoothed step
    v = v + H;
    s = s + H * (1.0 - H) / (0.75 * P.h);
  }
  volume_part[lid.x] = v; slope_part[lid.x] = s;
  var stride = 128u;
  loop {
    workgroupBarrier();
    if (lid.x < stride) {
      volume_part[lid.x] = volume_part[lid.x] + volume_part[lid.x + stride];
      slope_part[lid.x] = slope_part[lid.x] + slope_part[lid.x + stride];
    }
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  if (lid.x == 0u) {
    let cube = P.h * P.h * P.h;
    let volume = volume_part[0] * cube;
    let slope = slope_part[0] * cube;
    stats[0] = volume; stats[1] = slope;
    stats[2] = (target_volume[0] - volume) / max(slope, 1e-6);   // Newton step
  }
}`;

// add the Newton shift to every cell (the whole surface moves up or down)
export const SHIFT = /* wgsl */ `
${GRID}
@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read_write> phi: array<f32>;
${INDEX}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let index = threadIndex(gid, groups);
  if (index >= G.nx * G.ny * G.nz) { return; }
  phi[index] = phi[index] + stats[2];
}`;

// pressure = moving part + still-water part of the NEW surface;  eq. 8: no pressure in the air
export const MASK_PRESSURE = /* wgsl */ `
${GRID}
@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read> phi: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
${INDEX}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let index = threadIndex(gid, groups);
  if (index >= G.nx * G.ny * G.nz) { return; }
  pressure[index] = select(0.0, pressure[index] + max(phi[index], 0.0), phi[index] > 0.0);
}`;

// ---------------------------------------------------------------------------------------------------------------
// Eq. 6 built in: one re-distancing pseudo-step (Godunov upwind, as physical_equations.redistance). Inside cells only;
// the walls copy the inside afterwards (COPY_WALLS_FROM_INSIDE). sign_source = phi before this re-distancing.
// ---------------------------------------------------------------------------------------------------------------
export const REDISTANCE = /* wgsl */ `
${GRID}
@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read> phi: array<f32>;
@group(0) @binding(2) var<storage, read> sign_source: array<f32>;
@group(0) @binding(3) var<storage, read_write> phi_out: array<f32>;
${INDEX}
fn at(x: u32, y: u32, z: u32) -> f32 { return phi[(x * G.ny + y) * G.nz + z]; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let index = threadIndex(gid, groups);
  if (index >= G.nx * G.ny * G.nz) { return; }
  let x = index / (G.ny * G.nz); let y = (index / G.nz) % G.ny; let z = index % G.nz;
  let f = phi[index];
  if (x == 0u || y == 0u || z == 0u || x == G.nx - 1u || y == G.ny - 1u || z == G.nz - 1u) { phi_out[index] = f; return; }
  let s0 = sign_source[index];
  let s = s0 / sqrt(s0 * s0 + G.h * G.h);                      // smoothed sign
  let backward = vec3<f32>(f - at(x - 1u, y, z), f - at(x, y - 1u, z), f - at(x, y, z - 1u)) / G.h;
  let forward = vec3<f32>(at(x + 1u, y, z) - f, at(x, y + 1u, z) - f, at(x, y, z + 1u) - f) / G.h;
  var g2 = 0.0;
  for (var d = 0u; d < 3u; d = d + 1u) {
    let b = backward[d]; let fw = forward[d];
    let water_side = max(max(b, 0.0) * max(b, 0.0), min(fw, 0.0) * min(fw, 0.0));
    let air_side = max(min(b, 0.0) * min(b, 0.0), max(fw, 0.0) * max(fw, 0.0));
    g2 = g2 + select(air_side, water_side, s > 0.0);
  }
  phi_out[index] = f - 0.5 * G.h * s * (sqrt(g2) - 1.0);
}`;

// ---------------------------------------------------------------------------------------------------------------
// Velocity u = curl(A) on the cell faces (eq. 1), backward differences; the first cell copies itself (difference 0)
// ---------------------------------------------------------------------------------------------------------------
export const CURL = /* wgsl */ `
struct Params { nx: u32, ny: u32, nz: u32, p0: u32, h: f32, p1: f32, p2: f32, p3: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> potential: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<f32>;
${INDEX}
fn at(c: u32, x: u32, y: u32, z: u32) -> f32 {
  return potential[c * P.nx * P.ny * P.nz + (x * P.ny + y) * P.nz + z];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let cells = P.nx * P.ny * P.nz;
  let index = threadIndex(gid, groups);
  if (index >= cells) { return; }
  let x = index / (P.ny * P.nz); let y = (index / P.nz) % P.ny; let z = index % P.nz;
  let xm = select(x - 1u, 0u, x == 0u); let ym = select(y - 1u, 0u, y == 0u); let zm = select(z - 1u, 0u, z == 0u);
  let dAz_dy = (at(2u, x, y, z) - at(2u, x, ym, z)) / P.h;
  let dAy_dz = (at(1u, x, y, z) - at(1u, x, y, zm)) / P.h;
  let dAx_dz = (at(0u, x, y, z) - at(0u, x, y, zm)) / P.h;
  let dAz_dx = (at(2u, x, y, z) - at(2u, xm, y, z)) / P.h;
  let dAy_dx = (at(1u, x, y, z) - at(1u, xm, y, z)) / P.h;
  let dAx_dy = (at(0u, x, y, z) - at(0u, x, ym, z)) / P.h;
  velocity[0u * cells + index] = dAz_dy - dAy_dz;              // u_x
  velocity[1u * cells + index] = dAx_dz - dAz_dx;              // u_y
  velocity[2u * cells + index] = dAy_dx - dAx_dy;              // u_z
}`;
