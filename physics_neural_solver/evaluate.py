"""
evaluate.py  -  test a trained physics neural solver against textbook answers (checks C1-C5 of PLAN.md)

    python -m physics_neural_solver.evaluate --run coarse_small       # uses outputs/runs/coarse_small/checkpoint.pt

The analytical answers are used ONLY here, never in training. Each test runs the trained network on its own
for several seconds of water time (the same way the viewer will), and measures:

    still   water at rest stays at rest; pressure is hydrostatic; volume constant      (C1, C5)
    slosh   tilt to 10 degrees and hold: sloshing period vs omega^2 = g k tanh(k h)     (C2)
    sink    heavy stone (2500 kg/m^3, 7 cm) released under the surface: sinking speed   (C3)
    float   light stone (500 kg/m^3, 7 cm) released under water: fraction under water   (C4)
    roll    heavy stone on the floor, right end raised 10 degrees: it must roll left      (direction check)

Output: outputs/evaluation/<run>/results.json and summary.md
"""

import argparse                                                     # command-line options
import json                                                         # results file
import math                                                         # radians
from pathlib import Path                                            # file paths

import torch                                                        # tensors

from config import GRIDS, Scales                                    # grids and reference scales
from physics_neural_solver import physical_equations as eq          # equations and analytical checks
from physics_neural_solver.model import PhysicsNeuralSolver         # the network
from physics_neural_solver.simulation import Simulation             # scenes, tilt and the exact stone
from scene import water_volume                                      # volume of the water (eq. 7)


def load_model(run: str, device):
    checkpoint = torch.load(Path("outputs/runs") / run / "checkpoint.pt", map_location=device, weights_only=False)   # the trained network
    settings = checkpoint["settings"]                               # how it was trained
    model = PhysicsNeuralSolver(settings["network_size"]).to(device).eval()   # same architecture
    model.load_state_dict(checkpoint["model"])                      # trained weights
    return model, settings


class Rollout:
    """One scene stepped forward by the trained network (like the viewer)."""

    def __init__(self, model, settings, device, depth_m=0.25, tilt_deg=0.0):
        self.scales = Scales()                                      # reference scales
        self.model = model                                          # the trained network
        dt = settings["time_step_seconds"] / self.scales.time       # time step, dimensionless
        self.sim = Simulation(1, GRIDS[settings["grid"]], device, dt, self.scales)   # one scene
        self.index = torch.tensor([0], device=device)               # its index
        depth = torch.tensor([depth_m / self.scales.length], device=device)   # water depth, dimensionless
        tilt = torch.tensor([math.radians(tilt_deg)], device=device)   # box tilt
        self.sim.reset(self.index, depth, tilt)                     # calm water
        self.time = 0.0                                             # water time in seconds

    @torch.no_grad()
    def step(self):
        state, box = self.sim.inputs(self.index)                    # what the network reads
        next_state = self.model(state, box, self.sim.h)             # one network step
        self.sim.advance(self.index, next_state)                    # store it; move tilt and stone
        self.time += self.sim.dt * self.scales.time                 # seconds

    def surface_height_at(self, x_m):
        """Water surface height (metres above the floor) at x (metres from the box centre), middle of the width."""
        phi = self.sim.water.phi[0, 0]                              # (X, Y, Z)
        h = self.sim.h                                              # cell size, dimensionless
        half_x = self.sim.limits.half_size[0]                       # half the box length
        i = int(round((x_m / self.scales.length + half_x) / h + 0.5))   # column index
        k = phi.shape[2] // 2                                       # middle of the width
        column = phi[i, :, k]                                       # phi from the floor up
        wet = torch.nonzero(column[1:-1] > 0).flatten() + 1         # water cells
        if len(wet) == 0:
            return float("nan")                                     # no water in this column
        j = int(wet.max())                                          # highest water cell
        below, above = float(column[j]), float(column[j + 1])       # phi below and above the surface
        fraction = below / (below - above) if above <= 0 else 0.5   # where phi = 0
        y = (j - 0.5 + fraction) * h                                # height above the floor, dimensionless
        return y * self.scales.length                               # metres

    def max_speed(self):
        u = eq.faces_to_centres(eq.velocity_from_vector_potential(self.sim.water.potential, self.sim.h))   # velocity at the centres
        water = (self.sim.water.phi > 0) & (self.sim.solid == 0)    # only in the water
        return float(u.norm(dim=1, keepdim=True)[water].max()) * self.scales.speed   # m/s

    def volume_error(self):
        stone = torch.zeros_like(self.sim.water.phi)                # (volume of the water itself)
        volume = water_volume(self.sim.water.phi, self.sim.solid, stone, self.sim.h)
        return float((volume - self.sim.initial_volume) / self.sim.initial_volume)   # relative change


def test_still(model, settings, device, seconds=10.0):
    run = Rollout(model, settings, device)                          # calm water, level box
    worst_speed, worst_volume = 0.0, 0.0                            # largest values seen
    while run.time < seconds:
        run.step()
        worst_speed = max(worst_speed, run.max_speed())             # water should stay still
        worst_volume = max(worst_volume, abs(run.volume_error()))   # volume should stay constant
    phi, p = run.sim.water.phi[0, 0], run.sim.water.pressure[0, 0]  # final state
    water = (phi > 0) & (run.sim.solid[0, 0] == 0)                  # water cells
    pressure_error = float((p[water] - phi[water]).abs().mean())    # hydrostatic: p = depth (dimensionless)
    return {"max_water_speed_m_s": worst_speed, "max_volume_error": worst_volume,
            "mean_hydrostatic_pressure_error": pressure_error,
            "pass": worst_speed < 0.02 and worst_volume < 0.01}     # pass: < 2 cm/s and < 1 %


def test_slosh(model, settings, device, seconds=10.0):
    run = Rollout(model, settings, device)                          # level box
    run.sim.tilt_target[0] = math.radians(10.0)                     # tilt to 10 degrees and hold
    times, heights = [], []                                         # wall height over time
    while run.time < seconds:
        run.step()
        times.append(run.time)                                      # seconds
        heights.append(run.surface_height_at(0.45))                 # near the right-hand wall
    settled = [(t, y) for t, y in zip(times, heights) if t > 1.5 and y == y]   # after the tilt, valid heights
    mean = sum(y for _, y in settled) / max(len(settled), 1)        # average height
    crossings = [t0 + (t1 - t0) * (mean - y0) / (y1 - y0)           # upward zero crossings of (height - mean)
                 for (t0, y0), (t1, y1) in zip(settled, settled[1:]) if y0 < mean <= y1]
    period = (crossings[-1] - crossings[0]) / (len(crossings) - 1) if len(crossings) > 1 else float("nan")
    scales = Scales()                                               # reference scales
    theory = eq.check_sloshing_period(scales.box.length / scales.length, 1.0) * scales.time   # C2 in seconds
    return {"period_s": period, "theory_period_s": theory, "relative_error": abs(period - theory) / theory,
            "pass": abs(period - theory) / theory < 0.15}           # pass: within 15 %


def test_stone(model, settings, device, density, start_depth_m, seconds):
    run = Rollout(model, settings, device)                          # calm water, level box
    scales = Scales()                                               # reference scales
    radius = 0.035 / scales.length                                  # 7 cm stone
    surface_y = -scales.box.height / 2 / scales.length + 0.25 / scales.length   # water surface height (box frame)
    run.sim.throw_stone(0, (0.0, surface_y - start_depth_m / scales.length, 0.0), (0.0, 0.0, 0.0),
                        radius, density / scales.water.density)    # released at rest
    fastest_sinking = 0.0                                           # largest downward speed
    while run.time < seconds:
        run.step()
        fastest_sinking = max(fastest_sinking, -float(run.sim.stones.velocity[0, 1]) * scales.speed)   # m/s
    s = run.sim.stones                                              # final stone state
    depth = float(s.surface_height[0] - s.centre[0, 1])             # centre below the surface (dimensionless)
    under = float(eq.underwater_volume(radius, depth) / eq.stone_volume(radius))   # fraction under water
    return run, fastest_sinking, under


def test_sink(model, settings, device):
    run, fastest, _ = test_stone(model, settings, device, density=2500.0, start_depth_m=0.04, seconds=3.0)
    theory = eq.check_terminal_velocity(2500.0 / Scales().water.density, 0.035 / Scales().length) * Scales().speed   # C3
    on_floor = float(run.sim.stones.centre[0, 1]) < -1.0 + 0.035 / Scales().length + 0.01   # reached the bottom
    return {"max_sinking_speed_m_s": fastest, "theory_terminal_speed_m_s": theory, "reached_floor": on_floor,
            "pass": on_floor and 0.3 * theory < fastest < 1.2 * theory}   # sinks, at a believable speed


def test_float(model, settings, device):
    _, _, under = test_stone(model, settings, device, density=500.0, start_depth_m=0.08, seconds=6.0)
    theory = eq.check_floating_fraction(500.0 / Scales().water.density)   # C4
    return {"fraction_under_water": under, "theory_fraction": theory, "pass": abs(under - theory) < 0.15}


def test_roll(model, settings, device, seconds=6.0):
    run = Rollout(model, settings, device)                          # calm water, level box
    scales = Scales()                                               # reference scales
    radius = 0.035 / scales.length                                  # 7 cm stone
    floor_y = -scales.box.height / 2 / scales.length + radius + 0.002   # resting on the floor
    run.sim.throw_stone(0, (0.25 / scales.length, floor_y, 0.0), (0.0, 0.0, 0.0), radius, 2500.0 / scales.water.density)
    run.sim.tilt_target[0] = math.radians(10.0)                     # raise the right end
    start_x = float(run.sim.stones.centre[0, 0])                    # where it starts
    while run.time < seconds:
        run.step()
    end_x = float(run.sim.stones.centre[0, 0])                      # where it ends
    return {"start_x_m": start_x * scales.length, "end_x_m": end_x * scales.length,
            "pass": end_x < start_x - 0.1 / scales.length}          # rolled at least 10 cm to the left (downhill)


def main():
    parser = argparse.ArgumentParser(description="Check a trained solver against textbook answers (C1-C5).")
    parser.add_argument("--run", default="coarse_small", help="run name under outputs/runs/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="short versions of the tests (code check)")
    args = parser.parse_args()

    device = torch.device(args.device)                              # where to run
    model, settings = load_model(args.run, device)                  # the trained network
    seconds = 0.5 if args.quick else None                           # quick: very short tests
    results = {
        "still (C1, C5)": test_still(model, settings, device, **({"seconds": seconds} if seconds else {})),
        "slosh (C2)": test_slosh(model, settings, device, **({"seconds": seconds} if seconds else {})),
        "sink (C3)": test_sink(model, settings, device),
        "float (C4)": test_float(model, settings, device),
        "roll (direction)": test_roll(model, settings, device, **({"seconds": seconds} if seconds else {})),
    }

    out_dir = Path("outputs/evaluation") / args.run                 # where to write
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))   # all numbers
    lines = [f"# Evaluation of run '{args.run}'", ""]               # readable summary
    for name, result in results.items():
        lines.append(f"## {name}: {'PASS' if result['pass'] else 'FAIL'}")
        lines += [f"- {key}: {value:.4g}" if isinstance(value, float) else f"- {key}: {value}"
                  for key, value in result.items() if key != "pass"]
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
