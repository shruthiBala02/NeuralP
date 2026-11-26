// neuralP viewer: 3D water in a tilting box, simulated by the classical level-set solver (WebGPU)
// and drawn with Three.js. Later the same viewer will run the trained neural solver.
//
//   drag on the water's top (inside the box) = throw the stone like a slingshot
//   drag elsewhere = orbit the camera, scroll = zoom

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import GUI from "three/addons/libs/lil-gui.module.min.js";

import { BOX, GRIDS, STONE, WATER } from "./config.js";
import { WaterSolver } from "./solver/water_solver.js";
import { NeuralWaterSolver } from "./neural/neural_solver.js";
import { SurfaceMesher } from "./surface_nets.js";
import { Stone } from "./stone.js";

const message = document.getElementById("message");
function showError(text) {
  message.hidden = false;
  message.className = "error";
  message.textContent = text;
}
window.addEventListener("error", (e) => showError(`Error: ${e.message}`));
window.addEventListener("unhandledrejection", (e) => showError(`Error: ${e.reason?.message ?? e.reason}`));

// which solver:  ?solver=classical (default)   or   ?solver=neural&model=<run name>  (files from export_for_browser.py)
const query = new URLSearchParams(location.search);
const solverKind = query.get("solver") === "neural" ? "neural" : "classical";
const modelName = query.get("model") ?? "coarse_small";
let solver, gridName;
if (solverKind === "neural") {
  try {
    solver = await NeuralWaterSolver.create(`../outputs/export/${modelName}/`);   // the trained network
  } catch (error) {
    showError(`Could not load the neural solver "${modelName}" (${error.message}).\n` +
      "Export it first:  python -m physics_neural_solver.export_for_browser --run " + modelName + "\n" +
      "and serve the repository folder:  python3 -m http.server 8001  ->  http://localhost:8001/classical_solver/?solver=neural");
    throw error;
  }
  gridName = solver.manifest.grid.name;                                         // the grid the network was trained on
  if (query.get("test") === "equivalence") {                                    // browser vs PyTorch, one step
    solver.runEquivalenceTest().then((result) => {                              // (runs while the page keeps rendering)
      document.title = "equivalence " + Object.entries(result)
        .map(([name, r]) => typeof r === "string" ? `${name}: ${r}` : `${name}: max error ${r.maxError.toExponential(2)} (values up to ${r.maxValue.toFixed(3)})`).join(" | ");
    });
  }
} else {
  gridName = query.get("grid") === "fine" ? "fine" : "coarse";
  solver = await WaterSolver.create(GRIDS[gridName]);                           // the classical solver
}
message.hidden = true;

// ---------------------------------------------------------------- settings
const settings = {
  waterDepth: BOX.defaultDepth,
  tiltDegrees: 0,
  stoneDiameterCm: 7,
  stoneDensity: 2500,
  grid: gridName,
  solver: solverKind,
  resetWater: () => resetWater(),
};

// ---------------------------------------------------------------- scene
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: new URLSearchParams(location.search).has("test") });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe7ee);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.6;

const pivotHeight = BOX.standHeight + BOX.height / 2; // the box tilts about its centre
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.02, 50);
camera.position.set(1.35, 1.35, 1.55);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, pivotHeight - 0.05, 0);
controls.enableDamping = true;

const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(2, 4, 3);
scene.add(sun, new THREE.HemisphereLight(0xeef4ff, 0x7a7466, 0.8));

const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.MeshStandardMaterial({ color: 0xbfb8ad, roughness: 0.95 }));
floor.rotation.x = -Math.PI / 2;
scene.add(floor);

// stand: two posts holding an axle through the box centre (the tilt axis is z)
const standMaterial = new THREE.MeshStandardMaterial({ color: 0x55606a, roughness: 0.6, metalness: 0.3 });
for (const side of [-1, 1]) {
  const post = new THREE.Mesh(new THREE.BoxGeometry(0.06, pivotHeight, 0.04), standMaterial);
  post.position.set(0, pivotHeight / 2, side * (BOX.width / 2 + 0.06));
  const axle = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.06, 16), standMaterial);
  axle.rotation.x = Math.PI / 2;
  axle.position.set(0, pivotHeight, side * (BOX.width / 2 + 0.03));
  scene.add(post, axle);
}
const base = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.02, BOX.width + 0.25), standMaterial);
base.position.y = 0.01;
scene.add(base);

// everything in the box frame lives in boxGroup (origin = tilt pivot = box centre)
const boxGroup = new THREE.Group();
boxGroup.position.set(0, pivotHeight, 0);
scene.add(boxGroup);

const glass = new THREE.MeshPhysicalMaterial({ color: 0xffffff, transparent: true, opacity: 0.12, roughness: 0.05, depthWrite: false, side: THREE.DoubleSide });
const wall = 0.006;
for (const [sx, sy, sz, px, py, pz] of [
  [BOX.length + 2 * wall, wall, BOX.width + 2 * wall, 0, -BOX.height / 2 - wall / 2, 0],   // floor of the box
  [wall, BOX.height, BOX.width, -BOX.length / 2 - wall / 2, 0, 0],
  [wall, BOX.height, BOX.width, BOX.length / 2 + wall / 2, 0, 0],
  [BOX.length + 2 * wall, BOX.height, wall, 0, 0, -BOX.width / 2 - wall / 2],
  [BOX.length + 2 * wall, BOX.height, wall, 0, 0, BOX.width / 2 + wall / 2],
]) {
  const panel = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), glass);
  panel.position.set(px, py, pz);
  panel.renderOrder = 2;
  boxGroup.add(panel);
}
boxGroup.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(BOX.length, BOX.height, BOX.width)),
  new THREE.LineBasicMaterial({ color: 0x46525c })));

// water surface (rebuilt from phi after every solver frame)
const waterGeometry = new THREE.BufferGeometry();
const waterMaterial = new THREE.MeshPhysicalMaterial({ color: 0x2f86c8, roughness: 0.08, metalness: 0.0, transparent: true, opacity: 0.78 });
const waterMesh = new THREE.Mesh(waterGeometry, waterMaterial);
waterMesh.frustumCulled = false;
waterMesh.renderOrder = 1;
boxGroup.add(waterMesh);
const mesher = new SurfaceMesher(solver.nx, solver.ny, solver.nz, solver.h);

function updateWaterMesh(phi) {
  const surface = mesher.build(phi);
  waterGeometry.setAttribute("position", new THREE.BufferAttribute(surface.positions.slice(0, 3 * surface.vertexCount), 3));
  waterGeometry.setIndex(new THREE.BufferAttribute(surface.indices.slice(0, surface.indexCount), 1));
  waterGeometry.computeVertexNormals();
}

// the stone
const stone = new Stone();
const stoneMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshStandardMaterial({ color: 0x6f6a64, roughness: 0.85 }));
stoneMesh.visible = false;
boxGroup.add(stoneMesh);

// ---------------------------------------------------------------- water state
let tilt = 0, tiltRate = 0, tiltAcceleration = 0;
let currentGravity = [0, -WATER.gravity, 0]; // gravity in the box frame (eq. 3)
let phiShift = 0, volumeError = 0;

function resetWater() {
  stone.active = false;
  stoneMesh.visible = false;
  solver.reset(settings.waterDepth, tilt);
  updateWaterMesh(solver.latestPhi);
  phiShift = 0;
}
resetWater();

let previousPhi = null, simulationTime = 0;
const submittedTimes = [];
window.__series = [];
// how fast the surface moves (m/s, RMS over cells near the surface): used to check that the water settles
function surfaceSpeed(phi, previous, dt) {
  if (!previous || dt <= 0) return 0;
  const band = 1.5 * solver.h;
  let sum = 0, count = 0;
  for (let n = 0; n < phi.length; n++) {
    if (Math.abs(phi[n]) > band) continue;
    const v = (phi[n] - previous[n]) / dt;
    sum += v * v; count++;
  }
  return count ? Math.sqrt(sum / count) : 0;
}

function onSolverReadback(phi, velocity) {
  const [time, dt] = submittedTimes.shift() ?? [simulationTime, 0];
  window.__series.push([time, surfaceSpeed(phi, previousPhi, dt)]);
  if (testMode) {  // water height at the middle of the right-hand wall (for the sloshing period, check C2)
    const i = solver.nx - 2, k = Math.floor(solver.nz / 2);
    let height = null;
    for (let j = 1; j < solver.ny - 1; j++) {
      const a = phi[i + solver.nx * (j + solver.ny * k)], b = phi[i + solver.nx * (j + 1 + solver.ny * k)];
      if (a > 0 && b <= 0) { height = (j - 0.5 + a / (a - b)) * solver.h - BOX.height / 2; break; }
    }
    (window.__wallHeight ??= []).push([time, height]);
  }
  previousPhi = phi;
  updateWaterMesh(phi);
  stone.updateWaterState(solver, phi, velocity, currentGravity, dt);
  const volume = solver.waterVolume(phi, stone);
  const reference = solver.initialVolume;  // water is conserved; the stone only pushes the level up
  volumeError = (volume - reference) / reference;
  phiShift = 0.3 * (reference - volume) / (BOX.length * BOX.width); // global volume correction (eq. 7)
  solverFrames++;
}

// ---------------------------------------------------------------- throwing the stone (slingshot)
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const topSurface = new THREE.Mesh(new THREE.PlaneGeometry(BOX.length, BOX.width), new THREE.MeshBasicMaterial({ visible: false }));
topSurface.rotation.x = -Math.PI / 2;
topSurface.position.y = BOX.height / 2;
boxGroup.add(topSurface);
const aimArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.1, 0xd9480f, 0.04, 0.025);
aimArrow.visible = false;
boxGroup.add(aimArrow);
let aimStart = null;

function pointOnTopPlane(event) {
  pointer.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const normal = new THREE.Vector3(0, 1, 0).applyQuaternion(boxGroup.quaternion);
  const point = boxGroup.localToWorld(new THREE.Vector3(0, BOX.height / 2, 0));
  const hit = raycaster.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(normal, point), new THREE.Vector3());
  return hit ? boxGroup.worldToLocal(hit) : null;
}

function throwVelocity(start, end) {
  const pull = new THREE.Vector3().subVectors(start, end).multiplyScalar(5); // 20 cm pull = 1 m/s
  pull.y = 0;
  if (pull.length() > STONE.maxThrowSpeed) pull.setLength(STONE.maxThrowSpeed);
  return new THREE.Vector3(pull.x, 0, pull.z);  // released just under the lid; gravity does the rest
}

renderer.domElement.addEventListener("pointerdown", (event) => {
  pointer.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  if (raycaster.intersectObject(topSurface).length === 0) return;
  aimStart = pointOnTopPlane(event);
  controls.enabled = false;
});
renderer.domElement.addEventListener("pointermove", (event) => {
  if (!aimStart) return;
  const now = pointOnTopPlane(event);
  if (!now) return;
  const velocity = throwVelocity(aimStart, now);
  if (velocity.length() < 0.01) { aimArrow.visible = false; return; }  // a click without pulling = just drop it
  aimArrow.position.copy(aimStart);
  aimArrow.setDirection(velocity.clone().normalize());
  aimArrow.setLength(0.05 + 0.08 * velocity.length(), 0.04, 0.025);
  aimArrow.visible = true;
});
window.addEventListener("pointerup", (event) => {
  if (!aimStart) return;
  const end = pointOnTopPlane(event) ?? aimStart;
  const radius = settings.stoneDiameterCm / 200;
  const position = [
    THREE.MathUtils.clamp(aimStart.x, -BOX.length / 2 + radius, BOX.length / 2 - radius),
    BOX.height / 2 - radius - 0.005,  // just under the invisible lid
    THREE.MathUtils.clamp(aimStart.z, -BOX.width / 2 + radius, BOX.width / 2 - radius),
  ];
  stone.throw(position, throwVelocity(aimStart, end).toArray(), radius, settings.stoneDensity);
  stoneMesh.scale.setScalar(radius);
  stoneMesh.visible = true;
  aimStart = null;
  aimArrow.visible = false;
  controls.enabled = true;
});

// ---------------------------------------------------------------- controls panel
const gui = new GUI({ title: "Water box" });
gui.add(settings, "waterDepth", BOX.minDepth, BOX.maxDepth, 0.01).name("water level (m)").onFinishChange(resetWater);
gui.add(settings, "tiltDegrees", -BOX.maxTiltDegrees, BOX.maxTiltDegrees, 0.5).name("tilt (°): + raises right end");
gui.add(settings, "stoneDiameterCm", STONE.minDiameter * 100, STONE.maxDiameter * 100, 0.5).name("stone diameter (cm)");
gui.add(settings, "stoneDensity", STONE.minDensity, STONE.maxDensity, 50).name("stone density (kg/m³)");
gui.add(settings, "resetWater").name("reset water");
gui.add(settings, "solver", ["classical", "neural"]).name("solver").onChange((value) => {
  location.search = value === "neural" ? `?solver=neural&model=${modelName}` : `?grid=${gridName}`; });
if (solverKind === "classical") {
  gui.add(settings, "grid", Object.keys(GRIDS)).name("grid").onChange((value) => { location.search = `?grid=${value}`; });
}

// ---------------------------------------------------------------- animation loop
const timer = new THREE.Timer();
timer.connect(document);
let solverFrames = 0, solverStepsSinceUpdate = 0, statsClock = 0, stepsPerSecond = 0, framesPerSecond = 0, renderedFrames = 0;
let lastSubsteps = 1;
let totalRendered = 0;

// ?test=still|tilt|stone : headless self-test (fixed 1/60 s steps driven by a timer, scripted actions)
const testMode = new URLSearchParams(location.search).get("test");
let testTime = 0;
let testTimeSpeed = 1;
const testStone = { maxSinkSpeed: 0 };

function animate() {
  timer.update();
  let realDt = Math.min(timer.getDelta(), 1 / 30);
  if (testMode) {
    realDt = 1 / 60;
    testTime += realDt;
    testTimeSpeed = new URLSearchParams(location.search).get("speed") === "1" ? 1 : 3; // 3: 0.05 s of water time per test frame
    if ((testMode === "tilt" || testMode === "tilthold") && simulationTime > 0.2) settings.tiltDegrees = 10;
    const placeStone = (position, density) => {
      stone.throw(position, [0, 0, 0], 0.035, density);
      stoneMesh.scale.setScalar(0.035);
      stoneMesh.visible = true;
    };
    // roll: a heavy stone resting on the floor at the right; at 1.5 s the right end is raised 10 deg -> it must roll left
    if (testMode === "roll" && !stone.active && simulationTime > 0.05) placeStone([0.25, -BOX.height / 2 + 0.036, 0], 2500);
    if (testMode === "roll" && simulationTime > 1.5) settings.tiltDegrees = 10;
    // sink: a heavy stone released just under the surface (check C3: sinking speed)
    if (testMode === "sink" && !stone.active && simulationTime > 0.05) placeStone([0, -0.04, 0], 2500);
    // float: a light stone (500 kg/m3) released under water (check C4: half of it should end up under water)
    if (testMode === "float" && !stone.active && simulationTime > 0.05) placeStone([0, -0.08, 0], 500);
    if (stone.active) testStone.maxSinkSpeed = Math.max(testStone.maxSinkSpeed, -stone.velocity[1]);
    if (stone.active && totalRendered % 10 === 0) (window.__stoneLog ??= []).push(`${simulationTime.toFixed(1)}:${stone.position[1].toFixed(3)}/${stone.velocity[1].toFixed(2)}/${stone.submergedFraction.toFixed(2)}/${stone.waterAcceleration[1].toFixed(1)}`);
    if (testMode === "stone" && !stone.active && testTime > 0.3) {
      stone.throw([0.1, BOX.height / 2 - 0.04, 0.0], [0, -1.0, 0], 0.035, settings.stoneDensity);
      stoneMesh.scale.setScalar(0.035);
      stoneMesh.visible = true;
    }
  }
  const dt = solver.fixedStep ?? realDt * testTimeSpeed;   // the neural solver always steps by its training time step

  if (dt > 0 && query.get("test") !== "equivalence" && solver.readback.some((s) => !s.busy)) {   // (no simulation during the equivalence test)
    // tilt: move towards the slider angle, at most 30 degrees per second
    const target = THREE.MathUtils.degToRad(settings.tiltDegrees);
    const maxRate = THREE.MathUtils.degToRad(BOX.maxTiltRateDegrees);
    const newRate = THREE.MathUtils.clamp(4 * (target - tilt), -maxRate, maxRate);
    tiltAcceleration = (newRate - tiltRate) / dt;
    tiltRate = newRate;
    tilt += tiltRate * dt;
    const gravity = [-WATER.gravity * Math.sin(tilt), -WATER.gravity * Math.cos(tilt), 0]; // eq. 3
    currentGravity = gravity;

    const frame = { gravity, tiltRate, tiltAcceleration };
    stone.step(dt, frame);

    const speed = Math.max(1.0, Math.hypot(...stone.velocity) + 0.5);
    const substeps = THREE.MathUtils.clamp(Math.ceil(dt * speed / (0.7 * solver.h)), 1, 8);
    const subDt = dt / substeps;
    const stones = Array.from({ length: substeps }, (_, s) => (stone.active ? stone.predicted(s * subDt) : null));
    solver.runFrame({ ...frame, substeps, dt: subDt, phiShift, stones },
      onSolverReadback);
    simulationTime += dt;
    submittedTimes.push([simulationTime, dt]);
    phiShift = 0;
    lastSubsteps = substeps;
    solverStepsSinceUpdate += substeps;
  }

  boxGroup.rotation.z = tilt;
  if (stone.active) {
    stoneMesh.position.fromArray(stone.position);
    stoneMesh.quaternion.fromArray(stone.orientation);
  }
  controls.update();
  renderer.render(scene, camera);

  renderedFrames++;
  totalRendered++;
  if (testMode && totalRendered % 30 === 0) {  // headless screenshots miss canvas updates: mirror into an <img>
    let mirror = document.getElementById("testMirror");
    if (!mirror) {
      mirror = Object.assign(document.createElement("img"), { id: "testMirror" });
      Object.assign(mirror.style, { position: "absolute", left: "0", top: "0", width: "100%", height: "100%", pointerEvents: "none" });
      document.body.appendChild(mirror);
    }
    mirror.src = renderer.domElement.toDataURL();
  }
  if (testMode && testMode !== "equivalence" && totalRendered % 20 === 0) {
    const summary = [];
    for (let t = 0.5; t <= simulationTime; t += 1.0) {
      const near = window.__series.filter(([time]) => Math.abs(time - t) < 0.5).map(([, v]) => v);
      if (near.length) summary.push(`${t.toFixed(1)}s:${(near.reduce((a, b) => a + b, 0) / near.length * 1000).toFixed(1)}`);
    }
    // sloshing period from upward zero crossings of the wall height after the tilt has settled (t > 1 s)
    const samples = (window.__wallHeight ?? []).filter(([t, y]) => t > 1.0 && y !== null);
    const mean = samples.reduce((a, [, y]) => a + y, 0) / Math.max(1, samples.length);
    const crossings = [];
    for (let q = 1; q < samples.length; q++) {
      const [t0, y0] = samples[q - 1], [t1, y1] = samples[q];
      if (y0 - mean < 0 && y1 - mean >= 0) crossings.push(t0 + (t1 - t0) * (mean - y0) / (y1 - y0));
    }
    const period = crossings.length > 1 ? (crossings.at(-1) - crossings[0]) / (crossings.length - 1) : NaN;
    // water level away from the stone (column at x = -0.35 m), and how much of the stone is under water
    let level = NaN;
    const column = Math.round((-0.35 + BOX.length / 2) / solver.h + 0.5), kMid = Math.floor(solver.nz / 2);
    const phiNow = solver.latestPhi;
    for (let j = 1; j < solver.ny - 1; j++) {
      const a = phiNow[column + solver.nx * (j + solver.ny * kMid)], b = phiNow[column + solver.nx * (j + 1 + solver.ny * kMid)];
      if (a > 0 && b <= 0) { level = (j - 0.5 + a / (a - b)) * solver.h - BOX.height / 2; break; }
    }
    const r = stone.radius, capHeight = THREE.MathUtils.clamp(r + (level - stone.position[1]), 0, 2 * r);
    const underWater = (Math.PI * capHeight ** 2 * (3 * r - capHeight) / 3) / ((4 / 3) * Math.PI * r ** 3);
    const theory = Math.sqrt((8 / 3) * (2500 / WATER.density - 1) * WATER.gravity * r / 0.47);
    const stoneReport = stone.active
      ? `stone x=${stone.position[0].toFixed(3)} y=${stone.position[1].toFixed(3)} vx=${stone.velocity[0].toFixed(2)} vy=${stone.velocity[1].toFixed(2)} ` +
        `maxSink=${testStone.maxSinkSpeed.toFixed(2)} (theory terminal ${theory.toFixed(2)}) underwater=${underWater.toFixed(2)} ` +
        `Vfrac=${stone.submergedFraction.toFixed(2)} u=${stone.waterVelocity.map(v => v.toFixed(2))} dudt=${stone.waterAcceleration.map(v => v.toFixed(1))} log=${(window.__stoneLog ?? []).slice(-12).join(';')} | `
      : "";
    document.title = stoneReport + `period ${period.toFixed(3)}s (${crossings.length} crossings) | surface mm/s ${summary.join(" ")} | sim=${simulationTime.toFixed(2)} frames ${totalRendered} solverFrames ${solverFrames} vol ${(volumeError * 100).toFixed(2)}% ` +
      `tilt ${THREE.MathUtils.radToDeg(tilt).toFixed(1)} stone y=${stone.position[1].toFixed(3)} vy=${stone.velocity[1].toFixed(2)} wet=${stone.submergedFraction.toFixed(2)}`;
  }
  statsClock += realDt;
  if (statsClock > 0.5) {
    stepsPerSecond = solverStepsSinceUpdate / statsClock;
    framesPerSecond = renderedFrames / statsClock;
    solverStepsSinceUpdate = 0; renderedFrames = 0; statsClock = 0;
  }
}
if (testMode) setInterval(animate, 30);
else renderer.setAnimationLoop(animate);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
