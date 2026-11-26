// Builds the water's surface mesh from the level set phi (where phi = 0), with "naive surface nets":
// one vertex inside every grid cube that the surface passes through (the average of the points where
// the cube's edges cross phi = 0), and one quad for every grid edge that crosses the surface.
// Wall cells count as air, so the mesh is closed along the walls and the floor: you see a body of water.

import { BOX } from "./config.js";

const CORNERS = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];
const EDGES = [
  [0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7],
];

export class SurfaceMesher {
  constructor(nx, ny, nz, h) {
    Object.assign(this, { nx, ny, nz, h });
    this.vertexOfCube = new Int32Array((nx - 1) * (ny - 1) * (nz - 1));
    this.positions = new Float32Array(1 << 16);
    this.indices = new Uint32Array(1 << 17);
    this.half = [BOX.length / 2, BOX.height / 2, BOX.width / 2];
  }

  value(phi, i, j, k) {
    const { nx, ny, nz } = this;
    if (i === 0 || j === 0 || k === 0 || i === nx - 1 || j === ny - 1 || k === nz - 1) return -0.5 * this.h; // wall = air
    return phi[i + nx * (j + ny * k)];
  }

  // returns { positions, indices, vertexCount, indexCount } (arrays are reused between frames)
  build(phi, isStoneCell) {
    const { nx, ny, nz, h, half } = this;
    const cubeIndex = (i, j, k) => i + (nx - 1) * (j + (ny - 1) * k);
    this.vertexOfCube.fill(-1);
    let vertexCount = 0, indexCount = 0;
    const values = new Float32Array(8);

    for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      let mask = 0;
      for (let c = 0; c < 8; c++) {
        const [a, b, d] = CORNERS[c];
        values[c] = this.value(phi, i + a, j + b, k + d);
        if (values[c] > 0) mask |= 1 << c;
      }
      if (mask === 0 || mask === 255) continue;
      let sx = 0, sy = 0, sz = 0, crossings = 0;
      for (const [c0, c1] of EDGES) {
        const v0 = values[c0], v1 = values[c1];
        if ((v0 > 0) === (v1 > 0)) continue;
        const t = v0 / (v0 - v1);
        sx += CORNERS[c0][0] + t * (CORNERS[c1][0] - CORNERS[c0][0]);
        sy += CORNERS[c0][1] + t * (CORNERS[c1][1] - CORNERS[c0][1]);
        sz += CORNERS[c0][2] + t * (CORNERS[c1][2] - CORNERS[c0][2]);
        crossings++;
      }
      // cube corner (i, j, k) is the centre of cell (i, j, k)
      let x = (i + sx / crossings - 0.5) * h - half[0];
      let y = (j + sy / crossings - 0.5) * h - half[1];
      let z = (k + sz / crossings - 0.5) * h - half[2];
      x = Math.min(half[0], Math.max(-half[0], x));
      y = Math.min(half[1], Math.max(-half[1], y));
      z = Math.min(half[2], Math.max(-half[2], z));
      if (3 * vertexCount + 3 > this.positions.length) this.positions = grow(this.positions);
      this.positions.set([x, y, z], 3 * vertexCount);
      this.vertexOfCube[cubeIndex(i, j, k)] = vertexCount++;
    }

    const pushQuad = (a, b, c, d, flip) => {
      if (a < 0 || b < 0 || c < 0 || d < 0) return;
      if (indexCount + 6 > this.indices.length) this.indices = grow(this.indices);
      if (flip) this.indices.set([a, c, b, a, d, c], indexCount);
      else this.indices.set([a, b, c, a, c, d], indexCount);
      indexCount += 6;
    };
    const V = (i, j, k) => this.vertexOfCube[cubeIndex(i, j, k)];

    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const inside = this.value(phi, i, j, k) > 0;
      // edge along +x: quads made of the 4 cubes around it (in the y-z plane)
      if (i < nx - 1 && j >= 1 && k >= 1 && j < ny - 1 && k < nz - 1 && inside !== (this.value(phi, i + 1, j, k) > 0))
        pushQuad(V(i, j - 1, k - 1), V(i, j, k - 1), V(i, j, k), V(i, j - 1, k), !inside);
      if (j < ny - 1 && k >= 1 && i >= 1 && k < nz - 1 && i < nx - 1 && inside !== (this.value(phi, i, j + 1, k) > 0))
        pushQuad(V(i - 1, j, k - 1), V(i - 1, j, k), V(i, j, k), V(i, j, k - 1), !inside);
      if (k < nz - 1 && i >= 1 && j >= 1 && i < nx - 1 && j < ny - 1 && inside !== (this.value(phi, i, j, k + 1) > 0))
        pushQuad(V(i - 1, j - 1, k), V(i, j - 1, k), V(i, j, k), V(i - 1, j, k), !inside);
    }
    return { positions: this.positions, indices: this.indices, vertexCount, indexCount };
  }
}

function grow(array) {
  const bigger = new array.constructor(array.length * 2);
  bigger.set(array);
  return bigger;
}
