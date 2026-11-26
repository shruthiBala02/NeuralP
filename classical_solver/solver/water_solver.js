// The classical 3D water solver (level set on a MAC grid), running on the GPU with WebGPU compute.
//
// One solver step (all on the GPU):
//   classify cells -> eddy viscosity -> advect velocity -> gravity, tilt forces, viscosity
//   -> wall law -> wall / stone velocities -> pressure solve (red-black SOR) -> projection
//   -> wall / stone velocities -> extrapolate velocity into the air -> advect level set -> reinitialise
// After the last step of a frame, the level set phi and the velocity are copied back to the CPU
// (for drawing the surface, the volume correction and the water force on the stone).

import { BOX, WATER } from "../config.js";
import * as K from "./shaders.js";

const PARAM_BYTES = 144;   // size of the Params struct in shaders.js
const PARAM_STRIDE = 256;  // dynamic-offset alignment
const MAX_SUBSTEPS = 8;
const EXTRAPOLATION_PASSES = 4;   // must be even (the result ends in the original buffer)
const REINIT_ITERATIONS = 3;      // must be odd (the result ends in phi)
const REINIT_EVERY = 4;          // reinitialise only every few steps (each one nudges the surface)

export class WaterSolver {
  static async create(gridSettings) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser (use Chrome on the Mac).");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter found.");
    const device = await adapter.requestDevice();
    device.lost.then((info) => console.error("WebGPU device lost:", info.message));
    return new WaterSolver(device, gridSettings, adapter.info?.description || adapter.info?.vendor || "GPU");
  }

  constructor(device, gridSettings, adapterName) {
    this.device = device;
    this.adapterName = adapterName;
    this.settings = gridSettings;
    this.h = gridSettings.cellSize;
    this.nx = Math.round(BOX.length / this.h) + 2;
    this.ny = Math.round(BOX.height / this.h) + 2;
    this.nz = Math.round(BOX.width / this.h) + 2;
    this.cellCount = this.nx * this.ny * this.nz;
    this.workgroups = [Math.ceil(this.nx / 4), Math.ceil(this.ny / 4), Math.ceil(this.nz / 4)];

    this.createBuffers();
    this.createPipelines();
    this.readback = [this.createStaging(), this.createStaging()];
    this.paramData = new ArrayBuffer(PARAM_STRIDE * 2 * MAX_SUBSTEPS);
  }

  createBuffers() {
    const bytes = this.cellCount * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.buffers = {};
    for (const name of ["ux", "uy", "uz", "uxT", "uyT", "uzT", "p", "rhs", "phi", "phi0", "phiT", "nut", "cellType", "validA", "validB"]) {
      this.buffers[name] = this.device.createBuffer({ size: bytes, usage, label: name });
    }
    this.paramBuffer = this.device.createBuffer({
      size: PARAM_STRIDE * 2 * MAX_SUBSTEPS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "params",
    });
  }

  createStaging() {
    return {
      buffer: this.device.createBuffer({ size: this.cellCount * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      busy: false,
    };
  }

  createPipelines() {
    const device = this.device;
    this.paramsLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAM_BYTES } }],
    });
    this.paramsBindGroup = device.createBindGroup({
      layout: this.paramsLayout,
      entries: [{ binding: 0, resource: { buffer: this.paramBuffer, size: PARAM_BYTES } }],
    });

    const build = (spec) => {
      const storageLayout = device.createBindGroupLayout({
        entries: spec.bindings.map((b, index) => ({
          binding: index, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: b.access === "read" ? "read-only-storage" : "storage" },
        })),
      });
      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.paramsLayout, storageLayout] }),
        compute: { module: device.createShaderModule({ code: spec.code }), entryPoint: "main" },
      });
      const bindGroup = device.createBindGroup({
        layout: storageLayout,
        entries: spec.bindings.map((b, index) => ({ binding: index, resource: { buffer: this.buffers[b.buffer] } })),
      });
      return { pipeline, bindGroup };
    };

    const components = [0, 1, 2];
    const names = ["ux", "uy", "uz"];
    this.kernels = {
      classify: build(K.classifyKernel()),
      eddyViscosity: build(K.eddyViscosityKernel()),
      advectVelocity: build(K.advectVelocityKernel()),
      forces: components.map((c) => build(K.forcesKernel(c))),
      wallLaw: components.map((c) => build(K.wallLawKernel(c))),
      boundary: components.map((c) => build(K.boundaryVelocityKernel(c))),
      divergence: build(K.divergenceKernel()),
      pressure: build(K.pressureKernel()),
      project: components.map((c) => build(K.projectKernel(c))),
      markValid: components.map((c) => build(K.markValidKernel(c))),
      // per component: pass 0: u -> uT (A -> B), pass 1: uT -> u (B -> A), repeated
      extrapolate: components.map((c) => [
        build(K.extrapolateKernel(c, names[c], names[c] + "T", "validA", "validB")),
        build(K.extrapolateKernel(c, names[c] + "T", names[c], "validB", "validA")),
      ]),
      advectPhi: build(K.advectPhiKernel()),
      // reinitialisation: phi0 -> phi -> phiT -> phi ...
      reinitialise: [build(K.reinitialiseKernel("phi0", "phi")), build(K.reinitialiseKernel("phi", "phiT")),
                     build(K.reinitialiseKernel("phiT", "phi"))],
      copyPhi: build(K.copyKernel("phi0", "phi")),
    };
    this.stepCounter = 0;
  }

  // flat, calm water at depth `depth`, at right angles to gravity for tilt angle `tilt` (eqs. 14-16)
  reset(depth, tilt) {
    const { nx, ny, nz, h } = this;
    const phi = new Float32Array(this.cellCount);
    const pressure = new Float32Array(this.cellCount);
    const up = [Math.sin(tilt), Math.cos(tilt), 0];  // -gravity direction in the box frame
    const surfaceY = -BOX.height / 2 + depth;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const [x, y] = this.cellCentre(i, j, k);
      const value = up[0] * (0 - x) + up[1] * (surfaceY - y);
      const n = i + nx * (j + ny * k);
      phi[n] = value;
      pressure[n] = Math.max(value, 0) * WATER.density * WATER.gravity;
    }
    const zeros = new Float32Array(this.cellCount);
    const q = this.device.queue;
    q.writeBuffer(this.buffers.phi, 0, phi);
    q.writeBuffer(this.buffers.p, 0, pressure);
    for (const name of ["ux", "uy", "uz", "uxT", "uyT", "uzT", "nut", "rhs", "phi0", "phiT"]) q.writeBuffer(this.buffers[name], 0, zeros);
    this.latestPhi = phi;
    const zerosN = new Float32Array(this.cellCount);
    this.latestVelocity = { ux: zerosN, uy: zerosN, uz: zerosN };
    this.initialVolume = this.waterVolume(phi, null);
  }

  cellCentre(i, j, k) {
    const h = this.h;
    return [(i - 0.5) * h - BOX.length / 2, (j - 0.5) * h - BOX.height / 2, (k - 0.5) * h - BOX.width / 2];
  }

  // water volume from phi (smoothed step over ~1.5 cells), ignoring wall and stone cells (eq. 7)
  waterVolume(phi, stone) {
    const { nx, ny, nz, h } = this;
    const width = 1.5 * h;
    let volume = 0;
    for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
      const value = phi[i + nx * (j + ny * k)];
      if (stone && stone.containsCell(this.cellCentre(i, j, k))) continue;
      volume += Math.min(1, Math.max(0, 0.5 + value / (2 * width)));
    }
    return volume * h * h * h;
  }

  writeParams(slot, color, frame) {
    const view = new DataView(this.paramData, (slot * 2 + color) * PARAM_STRIDE, PARAM_STRIDE);
    const u32 = (offset, value) => view.setUint32(offset, value, true);
    const f32 = (offset, value) => view.setFloat32(offset, value, true);
    const v3 = (offset, v) => { f32(offset, v[0]); f32(offset + 4, v[1]); f32(offset + 8, v[2]); };
    u32(0, this.nx); u32(4, this.ny); u32(8, this.nz); u32(12, color);
    f32(16, this.h); f32(20, frame.dt); f32(24, WATER.density); f32(28, WATER.kinematicViscosity);
    v3(32, frame.gravity); f32(44, frame.tiltRate);
    f32(48, frame.tiltAcceleration); f32(52, WATER.smagorinskyConstant); f32(56, WATER.vonKarman); f32(60, WATER.logLawOffset);
    f32(64, 0.0); f32(68, 1.9);  // roughness 0 (smooth walls), SOR factor
    f32(72, slot === 0 ? frame.phiShift : 0); f32(76, BOX.length / 2);
    f32(80, BOX.height / 2); f32(84, BOX.width / 2);
    const stone = frame.stones[slot];
    f32(88, stone ? stone.radius : 0); f32(92, stone ? 1 : 0);
    v3(96, stone ? stone.position : [0, 0, 0]);
    v3(112, stone ? stone.velocity : [0, 0, 0]);
    v3(128, stone ? stone.spin : [0, 0, 0]);
  }

  // Run `frame.substeps` solver steps and read phi and p back. Returns false if the GPU is still busy.
  // frame: { substeps, dt, gravity, tiltRate, tiltAcceleration, phiShift, stones[substep] }
  runFrame(frame, onReadback) {
    const staging = this.readback.find((s) => !s.busy);
    if (!staging) return false;
    const substeps = Math.min(frame.substeps, MAX_SUBSTEPS);
    for (let s = 0; s < substeps; s++) { this.writeParams(s, 0, frame); this.writeParams(s, 1, frame); }
    this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramData, 0, PARAM_STRIDE * 2 * substeps);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    const run = (k, slot, color = 0) => {
      pass.setPipeline(k.pipeline);
      pass.setBindGroup(0, this.paramsBindGroup, [(slot * 2 + color) * PARAM_STRIDE]);
      pass.setBindGroup(1, k.bindGroup);
      pass.dispatchWorkgroups(...this.workgroups);
    };
    const K_ = this.kernels;
    for (let s = 0; s < substeps; s++) {
      run(K_.classify, s);
      run(K_.eddyViscosity, s);
      run(K_.advectVelocity, s);
      for (const k of K_.forces) run(k, s);
      for (const k of K_.wallLaw) run(k, s);
      for (const k of K_.boundary) run(k, s);
      run(K_.divergence, s);
      for (let it = 0; it < this.settings.pressureIterations; it++) { run(K_.pressure, s, 0); run(K_.pressure, s, 1); }
      for (const k of K_.project) run(k, s);
      for (const k of K_.boundary) run(k, s);
      for (let c = 0; c < 3; c++) {
        run(K_.markValid[c], s);
        for (let pass_ = 0; pass_ < EXTRAPOLATION_PASSES; pass_++) run(K_.extrapolate[c][pass_ % 2], s);
      }
      run(K_.advectPhi, s);
      if (this.stepCounter++ % REINIT_EVERY === 0) {
        for (let it = 0; it < REINIT_ITERATIONS; it++) run(K_.reinitialise[it], s);
      } else {
        run(K_.copyPhi, s);
      }
    }
    pass.end();
    const bytes = this.cellCount * 4;
    // read back: phi (surface, volume) and the velocity (for the water force on the stone)
    ["phi", "ux", "uy", "uz"].forEach((name, index) =>
      encoder.copyBufferToBuffer(this.buffers[name], 0, staging.buffer, index * bytes, bytes));
    this.device.queue.submit([encoder.finish()]);

    staging.busy = true;
    staging.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const data = new Float32Array(staging.buffer.getMappedRange());
      const N = this.cellCount;
      this.latestPhi = data.slice(0, N);
      this.latestVelocity = { ux: data.slice(N, 2 * N), uy: data.slice(2 * N, 3 * N), uz: data.slice(3 * N, 4 * N) };
      staging.buffer.unmap();
      staging.busy = false;
      onReadback(this.latestPhi, this.latestVelocity);
    }).catch((error) => { staging.busy = false; console.error(error); });
    return true;
  }
}
