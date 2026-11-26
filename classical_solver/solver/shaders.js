// WGSL compute shaders of the classical level-set water solver.
//
// Grid: staggered (MAC), attached to the box, one layer of wall cells on every side.
//   cell (i, j, k) centre:   ((i - 0.5) h - L/2, (j - 0.5) h - H/2, (k - 0.5) h - W/2)
//   ux[i,j,k] lives on the face between cells i and i+1 (same idea for uy, uz)
//   p, phi, cellType, nut live at the cell centres
//
// Every kernel is described by { code, bindings }. bindings[b] = { name, buffer, access, type }:
// the WGSL variable `name` at @group(1) @binding(b) is the solver buffer `buffer`.

const PARAMS = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, nz: u32, color: u32,
  h: f32, dt: f32, rho: f32, nu: f32,
  gravity: vec3<f32>, tiltRate: f32,
  tiltAcceleration: f32, cs: f32, kappa: f32, logLawOffset: f32,
  roughness: f32, sorOmega: f32, phiShift: f32, halfL: f32,
  halfH: f32, halfW: f32, stoneRadius: f32, stoneActive: f32,
  stoneCenter: vec3<f32>, pad1: f32,
  stoneVelocity: vec3<f32>, pad2: f32,
  stoneSpin: vec3<f32>, pad3: f32,
};
@group(0) @binding(0) var<uniform> P: Params;

const AIR: u32 = 0u;
const WATER: u32 = 1u;
const WALL: u32 = 2u;
const STONE: u32 = 3u;

fn idx(i: i32, j: i32, k: i32) -> u32 {
  return u32(i + i32(P.nx) * (j + i32(P.ny) * k));
}
fn inside(i: i32, j: i32, k: i32) -> bool {
  return i >= 0 && j >= 0 && k >= 0 && i < i32(P.nx) && j < i32(P.ny) && k < i32(P.nz);
}
fn clampedIdx(i: i32, j: i32, k: i32) -> u32 {
  return idx(clamp(i, 0, i32(P.nx) - 1), clamp(j, 0, i32(P.ny) - 1), clamp(k, 0, i32(P.nz) - 1));
}
fn isWallCell(i: i32, j: i32, k: i32) -> bool {
  return i == 0 || j == 0 || k == 0 || i == i32(P.nx) - 1 || j == i32(P.ny) - 1 || k == i32(P.nz) - 1;
}
fn halfSize() -> vec3<f32> { return vec3<f32>(P.halfL, P.halfH, P.halfW); }
fn cellCentre(i: i32, j: i32, k: i32) -> vec3<f32> {
  return (vec3<f32>(f32(i), f32(j), f32(k)) - 0.5) * P.h - halfSize();
}
fn unitStep(a: u32) -> vec3<i32> {
  if (a == 0u) { return vec3<i32>(1, 0, 0); }
  if (a == 1u) { return vec3<i32>(0, 1, 0); }
  return vec3<i32>(0, 0, 1);
}
fn pick(v: vec3<f32>, a: u32) -> f32 {
  if (a == 0u) { return v.x; }
  if (a == 1u) { return v.y; }
  return v.z;
}
fn pickIndex(v: vec3<i32>, a: u32) -> i32 {
  if (a == 0u) { return v.x; }
  if (a == 1u) { return v.y; }
  return v.z;
}
fn stoneVelocityAt(x: vec3<f32>) -> vec3<f32> {
  return P.stoneVelocity + cross(P.stoneSpin, x - P.stoneCenter);
}
`;

const KERNEL_START = /* wgsl */ `
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = i32(id.x);
  let j = i32(id.y);
  let k = i32(id.z);
  if (!inside(i, j, k)) { return; }
  let n = idx(i, j, k);
`;

function declarations(bindings) {
  return bindings.map((b, index) =>
    `@group(1) @binding(${index}) var<storage, ${b.access === "read" ? "read" : "read_write"}> ${b.name}: array<${b.type ?? "f32"}>;`
  ).join("\n");
}

function kernel(bindings, helpers, body) {
  return { code: PARAMS + declarations(bindings) + helpers + KERNEL_START + body + "\n}\n", bindings };
}

const read = (name, buffer = name, type = "f32") => ({ name, buffer, access: "read", type });
const write = (name, buffer = name, type = "f32") => ({ name, buffer, access: "write", type });
const COMPONENT_NAMES = ["ux", "uy", "uz"];

// trilinear sampling of `buffer` at fractional grid coordinates g (integer g = a stored sample)
function sampler(functionName, buffer) {
  return /* wgsl */ `
fn ${functionName}(g: vec3<f32>) -> f32 {
  let hi = vec3<f32>(f32(P.nx) - 1.001, f32(P.ny) - 1.001, f32(P.nz) - 1.001);
  let c = clamp(g, vec3<f32>(0.0), hi);
  let b = floor(c);
  let f = c - b;
  let i = i32(b.x);
  let j = i32(b.y);
  let k = i32(b.z);
  let x00 = mix(${buffer}[idx(i, j, k)], ${buffer}[idx(i + 1, j, k)], f.x);
  let x10 = mix(${buffer}[idx(i, j + 1, k)], ${buffer}[idx(i + 1, j + 1, k)], f.x);
  let x01 = mix(${buffer}[idx(i, j, k + 1)], ${buffer}[idx(i + 1, j, k + 1)], f.x);
  let x11 = mix(${buffer}[idx(i, j + 1, k + 1)], ${buffer}[idx(i + 1, j + 1, k + 1)], f.x);
  return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}`;
}

// velocity anywhere in the box (from the three staggered arrays) and a semi-Lagrangian backtrace
function velocityHelpers(ux, uy, uz) {
  return sampler("sampleUx", ux) + sampler("sampleUy", uy) + sampler("sampleUz", uz) + /* wgsl */ `
fn velocityAt(x: vec3<f32>) -> vec3<f32> {
  let g = (x + halfSize()) / P.h;
  return vec3<f32>(sampleUx(g + vec3<f32>(0.0, 0.5, 0.5)),
                   sampleUy(g + vec3<f32>(0.5, 0.0, 0.5)),
                   sampleUz(g + vec3<f32>(0.5, 0.5, 0.0)));
}
fn backtrace(x: vec3<f32>) -> vec3<f32> {
  let mid = x - 0.5 * P.dt * velocityAt(x);
  return x - P.dt * velocityAt(mid);
}`;
}

// helpers for a kernel that works on the faces of one velocity component A
function faceHelpers(component) {
  return /* wgsl */ `
const A: u32 = ${component}u;
fn otherCell(i: i32, j: i32, k: i32) -> vec3<i32> { return vec3<i32>(i, j, k) + unitStep(A); }
fn faceCentre(i: i32, j: i32, k: i32) -> vec3<f32> {
  return cellCentre(i, j, k) + 0.5 * P.h * vec3<f32>(unitStep(A));
}`;
}

// a face is "fluid" when it touches water and no wall or stone
const FLUID_FACE = /* wgsl */ `
fn isFluidFace(t0: u32, t1: u32) -> bool {
  return t0 != WALL && t1 != WALL && t0 != STONE && t1 != STONE && (t0 == WATER || t1 == WATER);
}`;


// fraction theta of the distance from a water cell centre to an air cell centre at which phi = 0
const GHOST_FLUID = /* wgsl */ `
fn surfaceFraction(phiWater: f32, phiAir: f32) -> f32 {
  return clamp(phiWater / (phiWater - phiAir + 1e-12), 0.05, 1.0);
}`;

// ---------------------------------------------------------------- kernels

// which cells are wall / stone / water / air
export function classifyKernel() {
  return kernel([read("phi"), write("cellType", "cellType", "u32")], "", /* wgsl */ `
  var t = AIR;
  if (isWallCell(i, j, k)) {
    t = WALL;
  } else if (P.stoneActive > 0.5 && distance(cellCentre(i, j, k), P.stoneCenter) < P.stoneRadius) {
    t = STONE;
  } else if (phi[n] > 0.0) {
    t = WATER;
  }
  cellType[n] = t;`);
}

// Smagorinsky eddy viscosity nu_t = (C_s h)^2 |S| (eq. 4)
export function eddyViscosityKernel() {
  return kernel([read("ux"), read("uy"), read("uz"), read("cellType", "cellType", "u32"), write("nut")], /* wgsl */ `
fn centreVelocity(i: i32, j: i32, k: i32) -> vec3<f32> {
  return 0.5 * vec3<f32>(ux[clampedIdx(i, j, k)] + ux[clampedIdx(i - 1, j, k)],
                         uy[clampedIdx(i, j, k)] + uy[clampedIdx(i, j - 1, k)],
                         uz[clampedIdx(i, j, k)] + uz[clampedIdx(i, j, k - 1)]);
}`, /* wgsl */ `
  if (cellType[n] != WATER) { nut[n] = 0.0; return; }
  let ddx = (centreVelocity(i + 1, j, k) - centreVelocity(i - 1, j, k)) / (2.0 * P.h);
  let ddy = (centreVelocity(i, j + 1, k) - centreVelocity(i, j - 1, k)) / (2.0 * P.h);
  let ddz = (centreVelocity(i, j, k + 1) - centreVelocity(i, j, k - 1)) / (2.0 * P.h);
  let s12 = 0.5 * (ddy.x + ddx.y);
  let s13 = 0.5 * (ddz.x + ddx.z);
  let s23 = 0.5 * (ddz.y + ddy.z);
  let s2 = ddx.x * ddx.x + ddy.y * ddy.y + ddz.z * ddz.z + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23);
  let strain = sqrt(2.0 * s2);
  nut[n] = (P.cs * P.h) * (P.cs * P.h) * strain;`);
}

// semi-Lagrangian advection of the velocity (u . grad u)
export function advectVelocityKernel() {
  return kernel([read("ux"), read("uy"), read("uz"), write("uxT"), write("uyT"), write("uzT")],
    velocityHelpers("ux", "uy", "uz"), /* wgsl */ `
  let c = cellCentre(i, j, k);
  let hh = 0.5 * P.h;
  let gx = (backtrace(c + vec3<f32>(hh, 0.0, 0.0)) + halfSize()) / P.h;
  let gy = (backtrace(c + vec3<f32>(0.0, hh, 0.0)) + halfSize()) / P.h;
  let gz = (backtrace(c + vec3<f32>(0.0, 0.0, hh)) + halfSize()) / P.h;
  uxT[n] = sampleUx(gx + vec3<f32>(0.0, 0.5, 0.5));
  uyT[n] = sampleUy(gy + vec3<f32>(0.5, 0.0, 0.5));
  uzT[n] = sampleUz(gz + vec3<f32>(0.5, 0.5, 0.0));`);
}

// gravity in the tilted box + tilt forces + viscosity (nu + nu_t) on fluid faces (eqs. 2-4)
export function forcesKernel(component) {
  const source = COMPONENT_NAMES[component] + "T";
  return kernel([read("uxT"), read("uyT"), read("uzT"), write("OUT", COMPONENT_NAMES[component]),
    read("nut"), read("cellType", "cellType", "u32")],
  velocityHelpers("uxT", "uyT", "uzT") + faceHelpers(component) + FLUID_FACE + /* wgsl */ `
fn faceOpen(i: i32, j: i32, k: i32) -> bool {
  let o = otherCell(i, j, k);
  if (!inside(i, j, k) || !inside(o.x, o.y, o.z)) { return false; }
  return cellType[idx(i, j, k)] != WALL && cellType[idx(o.x, o.y, o.z)] != WALL;
}
fn neighbourOrSelf(i: i32, j: i32, k: i32, own: f32) -> f32 {
  if (!faceOpen(i, j, k)) { return own; }  // zero gradient at walls: the wall law gives the wall friction
  return ${source}[idx(i, j, k)];
}`, /* wgsl */ `
  let o = otherCell(i, j, k);
  let u = ${source}[n];
  if (!inside(o.x, o.y, o.z)) { OUT[n] = u; return; }
  let no = idx(o.x, o.y, o.z);
  if (!isFluidFace(cellType[n], cellType[no])) { OUT[n] = u; return; }
  var lap = 0.0;
  for (var d = 0u; d < 3u; d = d + 1u) {
    let s = unitStep(d);
    lap = lap + neighbourOrSelf(i + s.x, j + s.y, k + s.z, u) + neighbourOrSelf(i - s.x, j - s.y, k - s.z, u) - 2.0 * u;
  }
  lap = lap / (P.h * P.h);
  let x = faceCentre(i, j, k);
  let v = velocityAt(x);
  let w = P.tiltRate;
  let force = P.gravity + vec3<f32>(
    P.tiltAcceleration * x.y + w * w * x.x + 2.0 * w * v.y,
    -P.tiltAcceleration * x.x + w * w * x.y - 2.0 * w * v.x,
    0.0);
  let viscosity = P.nu + 0.5 * (nut[n] + nut[no]);
  OUT[n] = u + P.dt * (pick(force, A) + viscosity * lap);`);
}

// wall law with roughness (eqs. 11-12): wall friction on the first water layer next to each wall
export function wallLawKernel(component) {
  const name = COMPONENT_NAMES[component];
  return kernel([write("OUT", name), read("cellType", "cellType", "u32")], faceHelpers(component) + FLUID_FACE + /* wgsl */ `
fn frictionVelocity(speed: f32, distance: f32) -> f32 {
  var uTau = sqrt(P.nu * speed / distance + 1e-12);
  for (var it = 0; it < 6; it = it + 1) {
    let yPlus = max(distance * uTau / P.nu, 11.0);
    let roughnessPlus = P.roughness * uTau / P.nu;
    let shift = log(1.0 + 0.3 * roughnessPlus) / P.kappa;
    uTau = max(speed / (log(yPlus) / P.kappa + P.logLawOffset - shift), 1e-7);
  }
  return uTau;
}`, /* wgsl */ `
  let o = otherCell(i, j, k);
  if (!inside(o.x, o.y, o.z)) { return; }
  if (!isFluidFace(cellType[n], cellType[idx(o.x, o.y, o.z)])) { return; }
  let cell = vec3<i32>(i, j, k);
  let counts = vec3<i32>(i32(P.nx), i32(P.ny), i32(P.nz));
  var u = OUT[n];
  for (var b = 0u; b < 3u; b = b + 1u) {
    if (b == A) { continue; }
    let c = pickIndex(cell, b);
    if (c == 1 || c == pickIndex(counts, b) - 2) {
      let speed = abs(u);
      if (speed > 1e-6) {
        let uTau = frictionVelocity(speed, 0.5 * P.h);
        let slowDown = min(speed, P.dt * uTau * uTau / P.h);
        u = u - sign(u) * slowDown;
      }
    }
  }
  OUT[n] = u;`);
}

// no flow through walls; water moves with the stone (eqs. 10, 13)
export function boundaryVelocityKernel(component) {
  const name = COMPONENT_NAMES[component];
  return kernel([write("OUT", name), read("cellType", "cellType", "u32")], faceHelpers(component), /* wgsl */ `
  let o = otherCell(i, j, k);
  if (!inside(o.x, o.y, o.z)) { return; }
  let t0 = cellType[n];
  let t1 = cellType[idx(o.x, o.y, o.z)];
  if (t0 == WALL || t1 == WALL) {
    OUT[n] = 0.0;
  } else if (t0 == STONE || t1 == STONE) {
    OUT[n] = pick(stoneVelocityAt(faceCentre(i, j, k)), A);
  }`);
}

// right-hand side of the pressure equation: (rho / dt) div u
export function divergenceKernel() {
  return kernel([read("ux"), read("uy"), read("uz"), read("cellType", "cellType", "u32"), write("rhs")], "", /* wgsl */ `
  if (cellType[n] != WATER) { rhs[n] = 0.0; return; }
  let div = (ux[n] - ux[idx(i - 1, j, k)] + uy[n] - uy[idx(i, j - 1, k)] + uz[n] - uz[idx(i, j, k - 1)]) / P.h;
  rhs[n] = P.rho / P.dt * div;`);
}

// one red-black SOR sweep of  sum(p_neighbour - p) / h^2 = rhs
// free surface (eq. 8): p = 0 exactly where phi = 0, between a water and an air cell (ghost fluid, Gibou et al. 2002)
export function pressureKernel() {
  return kernel([write("p"), read("rhs"), read("cellType", "cellType", "u32"), read("phi")], GHOST_FLUID, /* wgsl */ `
  if (u32((i + j + k) % 2) != P.color) { return; }
  if (cellType[n] != WATER) { p[n] = 0.0; return; }
  var total = 0.0;
  var count = 0.0;
  for (var d = 0u; d < 3u; d = d + 1u) {
    let s = unitStep(d);
    for (var side = -1; side <= 1; side = side + 2) {
      let m = idx(i + side * s.x, j + side * s.y, k + side * s.z);
      let t = cellType[m];
      if (t == WATER) { total = total + p[m]; count = count + 1.0; }
      else if (t == AIR) { count = count + 1.0 / surfaceFraction(phi[n], phi[m]); }  // p = 0 at the exact surface
    }
  }
  if (count == 0.0) { p[n] = 0.0; return; }
  let gaussSeidel = (total - P.h * P.h * rhs[n]) / count;
  p[n] = p[n] + P.sorOmega * (gaussSeidel - p[n]);`);
}

// subtract the pressure gradient: makes the water incompressible (eq. 1)
export function projectKernel(component) {
  const name = COMPONENT_NAMES[component];
  return kernel([write("OUT", name), read("p"), read("cellType", "cellType", "u32"), read("phi")],
    faceHelpers(component) + FLUID_FACE + GHOST_FLUID, /* wgsl */ `
  let o = otherCell(i, j, k);
  if (!inside(o.x, o.y, o.z)) { return; }
  let no = idx(o.x, o.y, o.z);
  let t0 = cellType[n];
  let t1 = cellType[no];
  if (!isFluidFace(t0, t1)) { return; }
  var gradient = 0.0;
  if (t0 == WATER && t1 == WATER) {
    gradient = (p[no] - p[n]) / P.h;
  } else if (t0 == WATER) {   // water -> air: p = 0 at the surface, a fraction theta of the way
    gradient = -p[n] / (surfaceFraction(phi[n], phi[no]) * P.h);
  } else {                    // air -> water
    gradient = p[no] / (surfaceFraction(phi[no], phi[n]) * P.h);
  }
  OUT[n] = OUT[n] - P.dt / P.rho * gradient;`);
}

// mark faces that have a trusted velocity (water, wall or stone faces)
export function markValidKernel(component) {
  return kernel([read("cellType", "cellType", "u32"), write("validOut", "validA", "u32")], faceHelpers(component), /* wgsl */ `
  let o = otherCell(i, j, k);
  if (!inside(o.x, o.y, o.z)) { validOut[n] = 0u; return; }
  let t0 = cellType[n];
  let t1 = cellType[idx(o.x, o.y, o.z)];
  let valid = t0 == WATER || t1 == WATER || t0 == WALL || t1 == WALL || t0 == STONE || t1 == STONE;
  validOut[n] = select(0u, 1u, valid);`);
}

// one pass of velocity extrapolation into the air just above the surface
export function extrapolateKernel(component, input, output, validIn, validOut) {
  return kernel([read("IN", input), write("OUT", output), read("VALID_IN", validIn, "u32"), write("VALID_OUT", validOut, "u32")],
    faceHelpers(component), /* wgsl */ `
  if (VALID_IN[n] == 1u) { OUT[n] = IN[n]; VALID_OUT[n] = 1u; return; }
  var total = 0.0;
  var count = 0.0;
  for (var d = 0u; d < 3u; d = d + 1u) {
    let s = unitStep(d);
    for (var side = -1; side <= 1; side = side + 2) {
      let a = i + side * s.x;
      let b = j + side * s.y;
      let c = k + side * s.z;
      if (inside(a, b, c)) {
        let m = idx(a, b, c);
        if (VALID_IN[m] == 1u) { total = total + IN[m]; count = count + 1.0; }
      }
    }
  }
  if (count > 0.0) { OUT[n] = total / count; VALID_OUT[n] = 1u; }
  else { OUT[n] = IN[n]; VALID_OUT[n] = 0u; }`);
}

// semi-Lagrangian level-set transport (eq. 5) + the global volume correction (eq. 7)
export function advectPhiKernel() {
  return kernel([read("ux"), read("uy"), read("uz"), read("phi"), write("phi0")],
    velocityHelpers("ux", "uy", "uz") + sampler("samplePhi", "phi"), /* wgsl */ `
  if (isWallCell(i, j, k)) { phi0[n] = phi[n]; return; }
  let x = backtrace(cellCentre(i, j, k));
  phi0[n] = samplePhi((x + halfSize()) / P.h + 0.5) + P.phiShift;`);
}

// one pseudo-time step of reinitialisation towards |grad phi| = 1 (eq. 6), Godunov upwinding
export function reinitialiseKernel(input, output) {
  return kernel([read("IN", input), read("SIGN", "phi0"), write("OUT", output)], "", /* wgsl */ `
  if (isWallCell(i, j, k)) {  // walls copy the nearest water/air cell (zero gradient)
    OUT[n] = IN[idx(clamp(i, 1, i32(P.nx) - 2), clamp(j, 1, i32(P.ny) - 2), clamp(k, 1, i32(P.nz) - 2))];
    return;
  }
  let f = IN[n];
  let s0 = SIGN[n];
  let s = s0 / sqrt(s0 * s0 + P.h * P.h);
  var g2 = 0.0;
  for (var d = 0u; d < 3u; d = d + 1u) {
    let e = unitStep(d);
    let backward = (f - IN[idx(i - e.x, j - e.y, k - e.z)]) / P.h;
    let forward = (IN[idx(i + e.x, j + e.y, k + e.z)] - f) / P.h;
    if (s > 0.0) {
      let a = max(backward, 0.0);
      let b = min(forward, 0.0);
      g2 = g2 + max(a * a, b * b);
    } else {
      let a = min(backward, 0.0);
      let b = max(forward, 0.0);
      g2 = g2 + max(a * a, b * b);
    }
  }
  OUT[n] = f - 0.5 * P.h * s * (sqrt(g2) - 1.0);`);
}

// copy one buffer into another (used on solver steps without reinitialisation)
export function copyKernel(input, output) {
  return kernel([read("IN", input), write("OUT", output)], "", /* wgsl */ `
  OUT[n] = IN[n];`);
}
