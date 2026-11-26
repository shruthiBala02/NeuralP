// All constants of the simple version (same values as PLAN.md and neuralP/neuralp/config.py).

export const WATER = {
  density: 998.2,              // kg/m^3, 20 C
  kinematicViscosity: 1.004e-6, // m^2/s
  gravity: 9.81,               // m/s^2
  smagorinskyConstant: 0.1,    // C_s (eq. 4)
  vonKarman: 0.41,             // kappa (eq. 11)
  logLawOffset: 5.2,           // B (eq. 11)
};

export const BOX = {
  length: 1.0,  // x, metres (inside)
  height: 0.5,  // y
  width: 0.5,   // z
  defaultDepth: 0.25,
  minDepth: 0.10,
  maxDepth: 0.35,
  maxTiltDegrees: 15,
  maxTiltRateDegrees: 30, // per second
  standHeight: 0.5,       // visual only
};

export const STONE = {
  minDiameter: 0.05,
  maxDiameter: 0.10,
  minDensity: 300,
  maxDensity: 3000,
  restitution: 0.3,
  friction: 0.4,
  maxThrowSpeed: 3.0, // m/s
};

// interior cells: coarse 40 x 20 x 20, fine 80 x 40 x 40 (+ one wall cell on every side)
export const GRIDS = {
  coarse: { cellSize: 0.025, pressureIterations: 120 },
  fine: { cellSize: 0.0125, pressureIterations: 200 },
};
