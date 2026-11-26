// The stone: a rigid sphere computed EXACTLY (Newton-Euler, eqs. 17-20), not by the water solver.
//
// It is integrated in the box frame (the same frame as the water grid), so the walls are simple
// planes. Because the box frame tilts, the stone also feels the tilt forces (the same ones as the
// water: Euler, centrifugal, Coriolis) - this is exactly equivalent to integrating in the room frame.
// The water force on the stone is the Morison equation (Morison et al. 1950), using the water
// surface (phi) and the water velocity u from the solver around the stone:
//   F_w = rho V_under g_up                       buoyancy (Archimedes, exact)
//       + rho V_under du/dt                      Froude-Krylov (the water's own acceleration)
//       + rho C_a V_under (du/dt - dV/dt)         added mass (C_a = 0.5 for a sphere)
//       + 1/2 rho C_d A |u - V| (u - V)           drag (C_d = 0.47 for a sphere)
// V_under = volume of the stone under water, A = its frontal area under water.

import { BOX, STONE, WATER } from "./config.js";

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const add = (a, b, s = 1) => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];

export class Stone {
  constructor() {
    this.active = false;
    this.position = [0, 0, 0];
    this.velocity = [0, 0, 0];
    this.spin = [0, 0, 0];
    this.orientation = [0, 0, 0, 1]; // quaternion (x, y, z, w), for drawing
    this.radius = 0.03;
    this.density = 2500;
    this.resetWaterState();
  }

  resetWaterState() {
    this.underwaterVolume = 0;
    this.submergedFraction = 0;
    this.waterVelocity = [0, 0, 0];      // local water velocity around the stone
    this.waterAcceleration = [0, 0, 0];  // its rate of change, du/dt
    this.previousWaterVelocity = null;
  }

  get volume() { return (4 / 3) * Math.PI * this.radius ** 3; }
  get mass() { return this.density * this.volume; }                  // eq. 19
  get inertia() { return 0.4 * this.mass * this.radius ** 2; }        // solid sphere

  throw(position, velocity, radius, density) {
    Object.assign(this, { active: true, position: [...position], velocity: [...velocity], spin: [0, 0, 0], radius, density });
    this.resetWaterState();
  }

  containsCell(centre) {
    if (!this.active) return false;
    const d = [centre[0] - this.position[0], centre[1] - this.position[1], centre[2] - this.position[2]];
    return Math.hypot(d[0], d[1], d[2]) < this.radius;
  }

  // Read what the Morison force needs from the solver (called after every solver frame):
  //  * the water surface height near the stone (found directly in 4 vertical columns around it)
  //    -> the volume under water (exact sphere cap)
  //  * the undisturbed water velocity u: averaged over water cells 2.5-3.5 stone radii away, where the
  //    stone's own disturbance has faded (it falls off like (radius/distance)^3), and its rate du/dt
  updateWaterState(solver, phi, velocity, gravity, frameDt) {
    if (!this.active) return;
    const { nx, ny, nz, h } = solver;
    const index = (x, axis) => Math.round((x + [BOX.length, BOX.height, BOX.width][axis] / 2) / h + 0.5);
    const clampIndex = (i, n) => Math.min(n - 2, Math.max(1, i));
    const r = this.radius;

    // 1. surface height in 4 columns around the stone
    const offset = r + 2 * h;
    let heightSum = 0, heights = 0;
    for (const [dx, dz] of [[offset, 0], [-offset, 0], [0, offset], [0, -offset]]) {
      const i = clampIndex(index(this.position[0] + dx, 0), nx);
      const k = clampIndex(index(this.position[2] + dz, 2), nz);
      for (let j = ny - 2; j >= 1; j--) {           // from the top down: the first water cell
        const below = phi[i + nx * (j + ny * k)];
        if (below <= 0) continue;
        const above = phi[i + nx * (j + 1 + ny * k)];
        const fraction = j + 1 <= ny - 2 && above <= 0 ? below / (below - above) : 0.5;
        heightSum += (j - 0.5 + fraction) * h - BOX.height / 2;
        heights++;
        break;
      }
    }
    this.surfaceHeight = heights > 0 ? heightSum / heights : -Infinity;  // box-frame y of the water surface
    this.updateUnderwaterVolume(gravity);

    // 2. undisturbed water velocity around the stone
    const inner = Math.max(2.5 * r, r + 2 * h), outer = Math.max(3.5 * r, r + 4 * h);
    const reach = Math.ceil(outer / h) + 1;
    const centre = this.position.map((x, axis) => index(x, axis));
    let sum = [0, 0, 0], count = 0;
    for (let k = centre[2] - reach; k <= centre[2] + reach; k++)
      for (let j = centre[1] - reach; j <= centre[1] + reach; j++)
        for (let i = centre[0] - reach; i <= centre[0] + reach; i++) {
          if (i < 2 || j < 2 || k < 2 || i > nx - 3 || j > ny - 3 || k > nz - 3) continue;
          const distance = Math.hypot(...add(solver.cellCentre(i, j, k), this.position, -1));
          if (distance < inner || distance > outer) continue;
          const n = i + nx * (j + ny * k);
          if (phi[n] <= 0) continue;
          sum = add(sum, [0.5 * (velocity.ux[n] + velocity.ux[n - 1]),
                          0.5 * (velocity.uy[n] + velocity.uy[n - nx]),
                          0.5 * (velocity.uz[n] + velocity.uz[n - nx * ny])]);
          count++;
        }
    const u = count > 0 ? scale(sum, 1 / count) : [0, 0, 0];
    if (this.previousWaterVelocity && frameDt > 0) {
      let rate = scale(add(u, this.previousWaterVelocity, -1), 1 / frameDt);
      const size = Math.hypot(...rate), limit = 2 * WATER.gravity;   // guard against read-back noise
      if (size > limit) rate = scale(rate, limit / size);
      this.waterAcceleration = add(scale(this.waterAcceleration, 0.7), rate, 0.3);  // smoothed du/dt
    }
    this.previousWaterVelocity = u;
    this.waterVelocity = u;
  }

  // volume under water from the stone's CURRENT position and the latest surface height (exact sphere cap).
  // Called at every small stone step, so buoyancy never lags behind the stone's motion.
  updateUnderwaterVolume(gravity) {
    const r = this.radius;
    const tiltCos = -gravity[1] / Math.hypot(...gravity);   // depth measured along "up"
    const depth = ((this.surfaceHeight ?? -Infinity) - this.position[1]) * tiltCos;
    const cap = Math.min(2 * r, Math.max(0, r + depth));
    this.underwaterVolume = Math.PI * cap * cap * (3 * r - cap) / 3;
    this.submergedFraction = this.underwaterVolume / this.volume;
  }

  // advance by dt (several small steps, for the contacts); frame = { gravity, tiltRate, tiltAcceleration }
  step(dt, frame) {
    if (!this.active) return;
    const substeps = 8;
    const h = dt / substeps;
    const m = this.mass;
    const rho = WATER.density;
    const g = Math.hypot(...frame.gravity);
    const up = scale(frame.gravity, -1 / g);
    const u = this.waterVelocity;
    const dudt = this.waterAcceleration;
    for (let s = 0; s < substeps; s++) {
      this.updateUnderwaterVolume(frame.gravity);
      const V = this.underwaterVolume;
      const addedMass = 0.5 * rho * V;                                   // C_a = 0.5 (sphere)
      const area = Math.PI * this.radius ** 2 * this.submergedFraction;  // frontal area under water
      const [x, y] = this.position;
      const w = frame.tiltRate, dw = frame.tiltAcceleration;
      const tiltForce = [
        m * (dw * y + w * w * x + 2 * w * this.velocity[1]),
        m * (-dw * x + w * w * y - 2 * w * this.velocity[0]),
        0,
      ];
      // forces that do not depend on the stone's acceleration
      let force = scale(frame.gravity, m);                        // weight
      force = add(force, scale(up, rho * g * V));                 // buoyancy
      force = add(force, scale(dudt, rho * V + addedMass));       // Froude-Krylov + added mass (water part)
      force = add(force, tiltForce);
      // (m + m_a) dV/dt = force + drag, with the drag treated implicitly for stability:
      //   drag = -k (V - u),  k = 1/2 rho C_d A |u - V|
      const relative = add(this.velocity, u, -1);
      const k = 0.5 * rho * 0.47 * area * Math.hypot(...relative);
      const effectiveMass = m + addedMass;
      const a = h * k / effectiveMass;
      const free = add(this.velocity, force, h / effectiveMass);  // velocity without drag
      this.velocity = scale(add(free, u, a), 1 / (1 + a));        // implicit drag towards the water velocity
      this.position = add(this.position, this.velocity, h);
      this.collideWithWalls();
      this.rotate(h);
    }
  }

  // bounce (restitution e) and slide (Coulomb friction mu) on the 6 box walls (eq. 20)
  collideWithWalls() {
    const half = [BOX.length / 2, BOX.height / 2, BOX.width / 2];
    for (let axis = 0; axis < 3; axis++) {
      for (const side of [-1, 1]) {
        const limit = half[axis] - this.radius;
        if (side * this.position[axis] <= limit) continue;
        this.position[axis] = side * limit;
        const normalSpeed = this.velocity[axis] * side;     // > 0: moving into the wall
        if (normalSpeed <= 0) continue;
        const normalImpulse = (1 + STONE.restitution) * normalSpeed;
        this.velocity[axis] -= side * normalImpulse;
        // friction against the sliding velocity of the contact point
        const normal = [0, 0, 0]; normal[axis] = side;
        const contact = add(this.velocity, cross(this.spin, scale(normal, this.radius)));
        const tangential = add(contact, normal, -(contact[0] * normal[0] + contact[1] * normal[1] + contact[2] * normal[2]));
        const slip = Math.hypot(...tangential);
        if (slip < 1e-9) continue;
        // a solid sphere's contact point reacts with effective mass m * 2/7
        const change = Math.min(STONE.friction * normalImpulse, slip * 2 / 7);
        const direction = scale(tangential, 1 / slip);
        this.velocity = add(this.velocity, direction, -change);
        const impulse = scale(direction, -change * this.mass);
        this.spin = add(this.spin, cross(scale(normal, this.radius), impulse), 1 / this.inertia);
      }
    }
  }

  rotate(h) {
    const [wx, wy, wz] = this.spin;
    const [qx, qy, qz, qw] = this.orientation;
    const dq = [
      0.5 * h * (wx * qw + wy * qz - wz * qy),
      0.5 * h * (wy * qw + wz * qx - wx * qz),
      0.5 * h * (wz * qw + wx * qy - wy * qx),
      0.5 * h * (-wx * qx - wy * qy - wz * qz),
    ];
    const q = this.orientation.map((v, i) => v + dq[i]);
    const length = Math.hypot(...q);
    this.orientation = q.map((v) => v / length);
  }

  // where the stone will be at time t into the frame (for the solver's substeps)
  predicted(t) {
    return { position: add(this.position, this.velocity, t), velocity: this.velocity, spin: this.spin, radius: this.radius };
  }
}
