// The physics neural solver in the browser: runs the exported 3D U-Net (physics_neural_solver/model.py) on the GPU,
// followed by the same STRICT RULES as PyTorch (exact, whatever the network outputs):
//   eq. 10  no flow through walls (A fixed on the wall planes)      eq. 5  the surface is carried by the flow
//   eq. 7   the water volume is restored exactly                   eq. 8  no pressure in the air
//   eq. 6   phi is re-distanced every 4 steps (as simulation.py)
//
// It has the same interface as the classical WaterSolver (reset, runFrame, latestPhi, latestVelocity, ...), so the
// viewer can switch between the two. One network step = one time step of `fixedStep` seconds (the training step).
// Inside, everything is dimensionless and in PyTorch's memory order (x slowest, z fastest); the read-back is converted
// to the viewer's order (x fastest) and to metres / metres per second for the stone and the surface mesh.

import { BOX, WATER } from "../config.js";
import * as S from "./shaders.js";

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;   // GPU arrays
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;                              // small parameter blocks

export class NeuralWaterSolver {
  // load model.json + weights.bin from `folder` (e.g. "../outputs/export/coarse_small/")
  static async create(folder) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser (use Chrome on the Mac).");
    const manifest = await (await fetch(folder + "model.json", { cache: "no-store" })).json();                 // network layout
    const weights = new Float32Array(await (await fetch(folder + "weights.bin", { cache: "no-store" })).arrayBuffer());   // all weights
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const device = await adapter.requestDevice({                                       // allow big activation buffers
      requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                        maxBufferSize: adapter.limits.maxBufferSize },
    });
    device.addEventListener("uncapturederror", (event) => {     // show GPU errors (e.g. a shader that does not compile)
      const text = `GPU error: ${event.error.message}`;
      console.error(text);
      document.title = text.slice(0, 500);
      window.dispatchEvent(new ErrorEvent("error", { message: text }));
    });
    const solver = new NeuralWaterSolver(device, manifest, weights, folder);
    solver.adapterName = adapter.info?.description || adapter.info?.vendor || "GPU";
    return solver;
  }

  constructor(device, manifest, weights, folder) {
    this.device = device;                                       // the GPU
    this.manifest = manifest;                                   // network layout and settings
    this.folder = folder;                                       // where the files came from
    this.fixedStep = manifest.time_step_s;                      // one network step, seconds
    this.scales = { length: manifest.scales.length_m,           // reference length (m)
                    speed: manifest.scales.speed_m_s,           // reference speed (m/s)
                    time: manifest.scales.time_s };             // reference time (s)
    this.h = manifest.grid.cell_size_m;                         // cell size, metres
    [this.nx, this.ny, this.nz] = manifest.grid.cells_with_walls;   // grid (with walls)
    this.cellCount = this.nx * this.ny * this.nz;
    this.hd = this.h / this.scales.length;                      // cell size, dimensionless
    const multiple = 2 ** manifest.network.depth;               // the U-Net halves the grid `depth` times
    this.padded = [this.nx, this.ny, this.nz].map((n) => Math.ceil(n / multiple) * multiple);   // padded grid
    this.layers = Object.fromEntries(manifest.layers.map((layer) => [layer.name, layer]));     // by name

    this.weightBuffer = this.storage(weights.length);           // all weights on the GPU
    device.queue.writeBuffer(this.weightBuffer, 0, weights);
    this.state = { phi: this.storage(this.cellCount), potential: this.storage(3 * this.cellCount),
                   pressure: this.storage(this.cellCount), velocity: this.storage(3 * this.cellCount) };   // the water
    this.pipelines = {};                                        // compiled shaders, by name
    this.operations = [];                                       // the network, as a list of GPU dispatches
    this.buildNetwork();
    this.readback = [this.staging(), this.staging()];           // two read-back buffers (one may be busy)
  }

  // ------------------------------------------------------------------------------------------- small helpers
  storage(count) { return this.device.createBuffer({ size: Math.max(16, count * 4), usage: STORAGE }); }
  staging() { return { buffer: this.device.createBuffer({ size: 4 * this.cellCount * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false }; }

  uniform(words) {                                              // a uniform buffer from 32-bit words (numbers or {f: value})
    const data = new ArrayBuffer(Math.ceil(words.length / 4) * 16);
    const view = new DataView(data);
    words.forEach((w, i) => (typeof w === "object" ? view.setFloat32(4 * i, w.f, true) : view.setUint32(4 * i, w, true)));
    const buffer = this.device.createBuffer({ size: data.byteLength, usage: UNIFORM });
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  pipeline(name, code) {                                        // compile a shader once
    if (!this.pipelines[name]) {
      this.pipelines[name] = this.device.createComputePipeline({
        layout: "auto", compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" } });
    }
    return this.pipelines[name];
  }

  operation(name, code, buffers, threads, workgroupsOverride, list = this.operations) {   // one dispatch: shader + buffers + threads
    const pipeline = this.pipeline(name, code);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const groups = workgroupsOverride ?? Math.ceil(threads / 64);   // 64 threads per workgroup
    const workgroups = workgroupsOverride ? [groups, 1, 1] : [Math.min(groups, 65535), Math.ceil(groups / 65535), 1];
    list.push({ pipeline, bindGroup, workgroups });
  }

  offset(name) { return this.layers[name].offset; }             // where a weight tensor starts in weights.bin

  // ------------------------------------------------------------------------------------------- the network
  buildNetwork() {
    const { base_channels: base, depth } = this.manifest.network;
    const channels = Array.from({ length: depth + 1 }, (_, level) => base * 2 ** level);   // e.g. 16, 32, 64, 128
    const sizes = Array.from({ length: depth + 1 }, (_, level) => this.padded.map((n) => n / 2 ** level));   // grid per level
    const cells = sizes.map(([x, y, z]) => x * y * z);

    this.inputParams = this.device.createBuffer({ size: 112, usage: UNIFORM });   // rewritten every step (stone, tilt)
    const input = this.storage(15 * cells[0]);                  // the 15 input channels
    this.debugBuffers = { input: [input, 15 * cells[0]] };      // intermediate results checked by the equivalence test
    this.operation("assemble", S.ASSEMBLE_INPUT, [this.inputParams, this.state.phi, this.state.potential,
                                                  this.state.pressure, input], cells[0]);

    const convolution = (source, cin, cout, level, prefix, k = 3) => {   // conv (+ GroupNorm + SiLU unless k = 1)
      const output = this.storage(cout * cells[level]);
      const [nx, ny, nz] = sizes[level];
      const params = this.uniform([cin, cout, nx, ny, nz, k, this.offset(prefix + ".weight"), this.offset(prefix + ".bias")]);
      this.operation("conv", S.CONVOLUTION, [params, source, this.weightBuffer, output], cout * cells[level]);
      return output;
    };
    const groupNorm = (data, c, level, prefix) => {             // GroupNorm(min(8, c)) + SiLU, in place
      const groups = Math.min(8, c);
      const statistics = this.storage(2 * groups);
      const params = this.uniform([c, groups, cells[level], this.offset(prefix + ".weight"), this.offset(prefix + ".bias"), 0, 0, 0]);
      this.operation("gn_stats", S.GROUP_NORM_STATISTICS, [params, data, statistics], 0, groups);
      this.operation("gn_apply", S.GROUP_NORM_APPLY, [params, data, statistics, this.weightBuffer], c * cells[level]);
    };
    const block = (source, cin, cout, level, prefix) => {       // ConvBlock: (conv, GN, SiLU) x 2
      const a = convolution(source, cin, cout, level, `${prefix}.layers.0`);
      groupNorm(a, cout, level, `${prefix}.layers.1`);
      const b = convolution(a, cout, cout, level, `${prefix}.layers.3`);
      groupNorm(b, cout, level, `${prefix}.layers.4`);
      return b;
    };

    let x = input, previous = 15;                               // going down
    const skips = [];
    for (let level = 0; level < depth; level++) {
      const features = block(x, previous, channels[level], level, `network.down_blocks.${level}`);
      skips.push(features);                                     // kept for the way up
      const pooled = this.storage(channels[level] * cells[level + 1]);
      const [px, py, pz] = sizes[level + 1];
      this.operation("pool", S.MAX_POOL, [this.uniform([channels[level], px, py, pz]), features, pooled], channels[level] * cells[level + 1]);
      x = pooled;
      previous = channels[level];
    }
    x = block(x, channels[depth - 1], channels[depth], depth, "network.bottom");   // coarsest level

    for (let i = 0; i < depth; i++) {                           // going up
      const level = depth - 1 - i;
      const merged = this.storage(2 * channels[level] * cells[level]);   // [upsampled, skip]
      const [nx, ny, nz] = sizes[level];
      const prefix = `network.up_samplers.${i}`;
      this.operation("up", S.UPSAMPLE, [this.uniform([channels[level + 1], channels[level], nx, ny, nz,
        this.offset(prefix + ".weight"), this.offset(prefix + ".bias"), 0]), x, this.weightBuffer, merged], channels[level] * cells[level]);
      const count = channels[level] * cells[level];
      this.operation("copy", S.COPY, [this.uniform([count, count, 0, 0]), skips[level], merged], count);   // skip after the upsampled half
      x = block(merged, 2 * channels[level], channels[level], level, `network.up_blocks.${i}`);
    }
    const change = convolution(x, channels[0], 4, 0, "network.output", 1);   // 1x1 conv to the 4 outputs (dA, dp)
    this.debugBuffers.change = [change, 4 * cells[0]];

    // ---- the strict rules (model.PhysicsNeuralSolver.forward), in the same order as PyTorch
    const N = this.cellCount;
    const [px, py, pz] = this.padded;
    const cropParams = this.uniform([this.nx, this.ny, this.nz, 0, px, py, pz, 0]);
    const grid = (axis = 0) => this.uniform([this.nx, this.ny, this.nz, axis, { f: this.hd },
                                             { f: this.fixedStep / this.scales.time }, { f: 0 }, { f: 0 }]);
    this.gridParams = grid();
    this.velocityNow = this.storage(3 * N);                     // u = curl(A) before the step
    this.phiTemporary = this.storage(N);                        // scratch for phi
    this.phiSign = this.storage(N);                             // phi before re-distancing (its sign)
    this.volumeStats = this.storage(4);                         // volume, d(volume)/d(shift), shift
    this.targetVolume = this.storage(4);                        // the water volume to keep (eq. 7)
    const curlParams = this.uniform([this.nx, this.ny, this.nz, 0, { f: this.hd }, { f: 0 }, { f: 0 }, { f: 0 }]);
    this.operation("curl", S.CURL, [curlParams, this.state.potential, this.velocityNow], N);   // u now
    const mean = this.storage(3);
    this.operation("mean", S.POTENTIAL_CHANGE_MEAN, [cropParams, change, mean], 0, 3);   // mean of dA (3 workgroups)
    this.operation("apply", S.APPLY_CHANGE, [cropParams, change, mean, this.state.potential, this.state.pressure,
                                             this.state.phi], N);   // A (walls), moving part of p
    this.operation("curl", S.CURL, [curlParams, this.state.potential, this.state.velocity], N);   // u next
    this.operation("carry", S.CARRY_SURFACE, [this.gridParams, this.state.phi, this.velocityNow, this.state.velocity,
                                              this.phiTemporary], N);   // eq. 5: phi carried by the flow
    this.copyWalls(this.phiTemporary, this.operations);         // walls copy the inside
    this.operation("copy", S.COPY, [this.uniform([N, 0, 0, 0]), this.phiTemporary, this.state.phi], N);
    this.restoreVolume(this.operations);                        // eq. 7: exactly the same water
    this.operation("mask_p", S.MASK_PRESSURE, [this.gridParams, this.state.phi, this.state.pressure], N);   // eq. 8

    // ---- every few steps: re-distancing (eq. 6, simulation.py), then the volume again
    this.redistanceOperations = [];
    const R = this.redistanceOperations;
    this.operation("copy", S.COPY, [this.uniform([N, 0, 0, 0]), this.state.phi, this.phiSign], N, undefined, R);
    for (let it = 0; it < this.manifest.strict_rules.redistance_iterations; it++) {
      this.operation("redistance", S.REDISTANCE, [this.gridParams, this.state.phi, this.phiSign, this.phiTemporary], N, undefined, R);
      this.copyWalls(this.phiTemporary, R);
      this.operation("copy", S.COPY, [this.uniform([N, 0, 0, 0]), this.phiTemporary, this.state.phi], N, undefined, R);
    }
    this.restoreVolume(R);

    // ---- measuring the volume after a reset
    this.measureOperations = [];
    this.operation("volume", S.VOLUME_SUM, [this.inputParams, this.state.phi, this.targetVolume, this.volumeStats], 0, 1,
                   this.measureOperations);
  }

  copyWalls(field, list) {                                      // x, then y, then z (as copy_walls_from_inside)
    for (let axis = 0; axis < 3; axis++) {
      this.operation("walls", S.COPY_WALLS_FROM_INSIDE, [this.uniform([this.nx, this.ny, this.nz, axis, { f: this.hd },
        { f: 0 }, { f: 0 }, { f: 0 }]), field], this.cellCount, undefined, list);
    }
  }

  restoreVolume(list) {                                         // Newton steps (as restore_volume)
    for (let it = 0; it < this.manifest.strict_rules.volume_newton_steps; it++) {
      this.operation("volume", S.VOLUME_SUM, [this.inputParams, this.state.phi, this.targetVolume, this.volumeStats], 0, 1, list);
      this.operation("shift", S.SHIFT, [this.gridParams, this.volumeStats, this.state.phi], this.cellCount, undefined, list);
    }
  }

  // ------------------------------------------------------------------------------------------- one step
  // params (dimensionless): { gravity[3], tiltRate, tiltAcceleration, stone: {active, centre[3], velocity[3], spin[3], radius} }
  writeInputParams(params) {
    const data = new ArrayBuffer(112);
    const u = new Uint32Array(data), f = new Float32Array(data);
    const stone = params.stone;
    u.set([this.nx, this.ny, this.nz, stone?.active ? 1 : 0, ...this.padded, 0], 0);
    f.set([this.hd, BOX.length / 2 / this.scales.length, BOX.height / 2 / this.scales.length, BOX.width / 2 / this.scales.length], 8);
    f.set([...params.gravity, params.tiltRate], 12);
    f.set([...(stone?.centre ?? [0, 0, 0]), params.tiltAcceleration], 16);
    f.set([...(stone?.velocity ?? [0, 0, 0]), stone?.radius ?? 0], 20);
    f.set([...(stone?.spin ?? [0, 0, 0]), 0], 24);
    this.device.queue.writeBuffer(this.inputParams, 0, data);
  }

  encodeStep(encoder, operations = this.operations) {
    const pass = encoder.beginComputePass();
    for (const op of operations) {                              // the whole network + strict rules, in order
      pass.setPipeline(op.pipeline);
      pass.setBindGroup(0, op.bindGroup);
      pass.dispatchWorkgroups(...op.workgroups);
    }
    pass.end();
  }

  // ------------------------------------------------------------------------------------------- viewer interface
  // calm water (eqs. 14-16) at `depth` metres with the box tilted by `tilt` radians
  reset(depth, tilt) {
    const N = this.cellCount, L = this.scales.length;
    const phi = new Float32Array(N), pressure = new Float32Array(N);
    const up = [Math.sin(tilt), Math.cos(tilt)];                // "up" in the box frame
    const surfaceY = -BOX.height / 2 / L + depth / L;            // surface point (dimensionless), box middle
    const inside = (a, n) => Math.min(Math.max(a, 1), n - 2);  // wall cells copy the nearest inside cell
    for (let i = 0; i < this.nx; i++) for (let j = 0; j < this.ny; j++) for (let k = 0; k < this.nz; k++) {
      const x = ((inside(i, this.nx) - 0.5) * this.h - BOX.length / 2) / L, y = ((inside(j, this.ny) - 0.5) * this.h - BOX.height / 2) / L;
      const t = (i * this.ny + j) * this.nz + k;                // PyTorch order
      phi[t] = up[0] * (0 - x) + up[1] * (surfaceY - y);        // signed distance below the surface
      pressure[t] = Math.max(phi[t], 0);                        // hydrostatic (dimensionless)
    }
    const q = this.device.queue;
    q.writeBuffer(this.state.phi, 0, phi);
    q.writeBuffer(this.state.pressure, 0, pressure);
    q.writeBuffer(this.state.potential, 0, new Float32Array(3 * N));   // at rest
    this.writeInputParams({ gravity: [0, -1, 0], tiltRate: 0, tiltAcceleration: 0, stone: null });   // no stone
    const encoder = this.device.createCommandEncoder();          // measure the starting volume on the GPU (eq. 7) ...
    this.encodeStep(encoder, this.measureOperations);
    encoder.copyBufferToBuffer(this.volumeStats, 0, this.targetVolume, 0, 4);   // ... and keep it as the target
    q.submit([encoder.finish()]);
    this.stepsSinceReset = 0;                                   // for re-distancing every few steps
    this.latestPhi = this.toViewerOrder(phi, L);
    const zeros = new Float32Array(N);
    this.latestVelocity = { ux: zeros, uy: zeros, uz: zeros };
    this.initialVolume = this.waterVolume(this.latestPhi, null);
  }

  // one network step. frame (metres, seconds, radians): { gravity[3] (m/s^2), tiltRate, tiltAcceleration, stones[0] }
  runFrame(frame, onReadback) {
    const staging = this.readback.find((s) => !s.busy);
    if (!staging) return false;                                 // the GPU is still busy with the last step
    const { length: L, speed: U, time: T } = this.scales;
    const stone = frame.stones?.[0];
    this.writeInputParams({
      gravity: frame.gravity.map((g) => g / WATER.gravity),     // |g| = 1
      tiltRate: frame.tiltRate * T, tiltAcceleration: frame.tiltAcceleration * T * T,
      stone: stone ? { active: true, centre: stone.position.map((v) => v / L), velocity: stone.velocity.map((v) => v / U),
                       spin: stone.spin.map((w) => w * T), radius: stone.radius / L } : null,
    });
    const encoder = this.device.createCommandEncoder();
    this.encodeStep(encoder);                                   // network + strict rules
    this.stepsSinceReset += 1;
    if (this.stepsSinceReset % this.manifest.strict_rules.redistance_every === 0) {
      this.encodeStep(encoder, this.redistanceOperations);      // eq. 6, every few steps (as simulation.py)
    }
    const bytes = this.cellCount * 4;
    encoder.copyBufferToBuffer(this.state.phi, 0, staging.buffer, 0, bytes);
    encoder.copyBufferToBuffer(this.state.velocity, 0, staging.buffer, bytes, 3 * bytes);
    this.device.queue.submit([encoder.finish()]);
    staging.busy = true;
    staging.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const data = new Float32Array(staging.buffer.getMappedRange()).slice();
      staging.buffer.unmap();
      staging.busy = false;
      const N = this.cellCount;
      this.latestPhi = this.toViewerOrder(data.subarray(0, N), L);   // metres
      this.latestVelocity = { ux: this.toViewerOrder(data.subarray(N, 2 * N), U),   // m/s
                              uy: this.toViewerOrder(data.subarray(2 * N, 3 * N), U),
                              uz: this.toViewerOrder(data.subarray(3 * N, 4 * N), U) };
      onReadback(this.latestPhi, this.latestVelocity);
    }).catch((error) => { staging.busy = false; console.error(error); });
    return true;
  }

  toViewerOrder(torchArray, scale) {                            // PyTorch order (z fastest) -> viewer order (x fastest), scaled
    const out = new Float32Array(this.cellCount);
    for (let i = 0; i < this.nx; i++) for (let j = 0; j < this.ny; j++) for (let k = 0; k < this.nz; k++) {
      out[i + this.nx * (j + this.ny * k)] = torchArray[(i * this.ny + j) * this.nz + k] * scale;
    }
    return out;
  }

  cellCentre(i, j, k) {                                         // metres, from the box centre
    return [(i - 0.5) * this.h - BOX.length / 2, (j - 0.5) * this.h - BOX.height / 2, (k - 0.5) * this.h - BOX.width / 2];
  }

  waterVolume(phi, stone) {                                     // m^3, the same rule as the classical solver
    const width = 1.5 * this.h;
    let volume = 0;
    for (let k = 1; k < this.nz - 1; k++) for (let j = 1; j < this.ny - 1; j++) for (let i = 1; i < this.nx - 1; i++) {
      if (stone && stone.containsCell(this.cellCentre(i, j, k))) continue;
      volume += Math.min(1, Math.max(0, 0.5 + phi[i + this.nx * (j + this.ny * k)] / (2 * width)));
    }
    return volume * this.h ** 3;
  }

  // ------------------------------------------------------------------------------------------- equivalence test
  // Runs the exported test case (one step on a known scene) and compares with PyTorch's answer.
  async runEquivalenceTest() {
    const tc = this.manifest.test_case;
    const data = new Float32Array(await (await fetch(this.folder + "test_case.bin", { cache: "no-store" })).arrayBuffer());
    const N = this.cellCount;
    const take = (() => { let at = 0; return (count) => data.subarray(at, (at += count)); })();
    const phi = take(N), potential = take(3 * N), pressure = take(N);
    const expected = { phi: take(N), potential: take(3 * N), pressure: take(N), velocity: take(3 * N) };
    expected.input = take(this.debugBuffers.input[1]);          // intermediate results
    expected.change = take(this.debugBuffers.change[1]);
    const expectedRedistanced = take(N);                        // phi after re-distancing + volume
    const q = this.device.queue;
    q.writeBuffer(this.state.phi, 0, phi);
    q.writeBuffer(this.state.potential, 0, potential);
    q.writeBuffer(this.state.pressure, 0, pressure);
    q.writeBuffer(this.targetVolume, 0, new Float32Array([tc.initial_volume, 0, 0, 0]));   // the volume to keep
    const gravity = [-Math.sin(tc.tilt), -Math.cos(tc.tilt), 0];
    this.writeInputParams({ gravity, tiltRate: tc.tilt_rate, tiltAcceleration: tc.tilt_acceleration, stone: tc.stone });
    const encoder = this.device.createCommandEncoder();
    this.encodeStep(encoder);
    const readBuffers = {};
    const sources = { phi: [this.state.phi, N], potential: [this.state.potential, 3 * N], pressure: [this.state.pressure, N],
                      velocity: [this.state.velocity, 3 * N], input: this.debugBuffers.input, change: this.debugBuffers.change };
    for (const [name, [source, count]] of Object.entries(sources)) {
      readBuffers[name] = this.device.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      encoder.copyBufferToBuffer(source, 0, readBuffers[name], 0, count * 4);
    }
    q.submit([encoder.finish()]);
    const result = {};
    for (const [name, buffer] of Object.entries(readBuffers)) {
      await buffer.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(buffer.getMappedRange());
      let worst = 0, size = 0;
      for (let i = 0; i < got.length; i++) {
        worst = Math.max(worst, Math.abs(got[i] - expected[name][i]));
        size = Math.max(size, Math.abs(expected[name][i]));
      }
      result[name] = { maxError: worst, maxValue: size };
      if (name === "input") {                                   // per channel, to see which one differs
        const per = got.length / 15;
        result.inputChannels = Array.from({ length: 15 }, (_, c) => {
          let w = 0, where = -1;
          for (let i = c * per; i < (c + 1) * per; i++) { const e = Math.abs(got[i] - expected.input[i]); if (e > w) { w = e; where = i - c * per; } }
          let gotSum = 0, expSum = 0;
          for (let i = c * per; i < (c + 1) * per; i++) { gotSum += got[i]; expSum += expected.input[i]; }
          return `${c}:${w.toExponential(1)}@${where}(sum ${gotSum.toFixed(2)} vs ${expSum.toFixed(2)})`;
        }).join(" ");
      }
      buffer.unmap();
    }
    const encoder2 = this.device.createCommandEncoder();         // then re-distancing (eq. 6) + volume, as every 4th step
    this.encodeStep(encoder2, this.redistanceOperations);
    const phiBuffer = this.device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder2.copyBufferToBuffer(this.state.phi, 0, phiBuffer, 0, N * 4);
    q.submit([encoder2.finish()]);
    await phiBuffer.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(phiBuffer.getMappedRange());
    let worst = 0, size = 0;
    for (let i = 0; i < N; i++) { worst = Math.max(worst, Math.abs(got[i] - expectedRedistanced[i])); size = Math.max(size, Math.abs(expectedRedistanced[i])); }
    result.redistanced = { maxError: worst, maxValue: size };
    phiBuffer.unmap();
    return result;
  }
}
