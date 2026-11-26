"""
Tests of Section 5 (the exact stone) and of its integration in simulation.py. The water is held calm (no network),
so the stone's physics is tested on its own: heavy stones sink and rest on the floor, light stones float at the
right depth, stones roll downhill when the box is tilted, and they never leave the box.
"""

import math                                                         # pi, radians

import pytest                                                       # parametrised tests
import torch                                                        # tensors

from config import COARSE, Scales                                   # grid and reference scales
from physics_neural_solver import physical_equations as eq          # stone equations
from physics_neural_solver.simulation import Simulation             # the stone integrator

SCALES = Scales()                                                   # reference scales
RADIUS = 0.035 / SCALES.length                                      # a 7 cm stone, dimensionless


# ---------------------------------------------------------------------------------------------- formulas
def test_mass_inertia_volume():
    r = torch.tensor(0.2)
    assert torch.isclose(eq.stone_volume(r), torch.tensor(4 / 3 * math.pi * 0.008))
    assert torch.isclose(eq.stone_mass(torch.tensor(2.5), r), 2.5 * eq.stone_volume(r))   # m = rho_s V
    assert torch.isclose(eq.stone_inertia(torch.tensor(3.0), r), torch.tensor(0.4 * 3.0 * 0.04))   # I = 2/5 m a^2


def test_underwater_volume_limits():
    r = torch.tensor(0.2)
    full = eq.stone_volume(r)
    assert float(eq.underwater_volume(r, torch.tensor(-0.3))) == 0.0                  # above the water
    assert torch.isclose(eq.underwater_volume(r, torch.tensor(0.0)), full / 2)        # half under
    assert torch.isclose(eq.underwater_volume(r, torch.tensor(0.5)), full)            # fully under
    depths = torch.linspace(-0.2, 0.2, 21)                                            # moving down through the surface
    volumes = eq.underwater_volume(r.expand(21), depths)
    assert torch.all(volumes[1:] >= volumes[:-1])                                     # grows smoothly, never jumps back


def test_morison_buoyancy_balances_weight_at_the_floating_depth():
    density = torch.tensor(0.5)                                     # half as dense as water
    r = torch.tensor(RADIUS)
    mass = eq.stone_mass(density, r)
    gravity = torch.tensor([0.0, -1.0, 0.0])
    volume_under = 0.5 * eq.stone_volume(r)                         # half under water (= rho_s / rho)
    force, _, _ = eq.morison_force(volume_under, r, gravity, torch.zeros(3), torch.zeros(3), torch.zeros(3), mass)
    assert torch.allclose(force + mass * gravity, torch.zeros(3), atol=1e-7)   # buoyancy + weight = 0: floats


def test_morison_drag_matches_the_terminal_velocity_formula():
    density, r = 2.5, torch.tensor(RADIUS)
    v_t = eq.check_terminal_velocity(density, float(r))             # C3
    mass = eq.stone_mass(torch.tensor(density), r)
    full = eq.stone_volume(r)
    _, k, _ = eq.morison_force(full, r, torch.tensor([0.0, -1.0, 0.0]), torch.zeros(3), torch.zeros(3),
                               torch.tensor([0.0, -v_t, 0.0]), mass)
    net_weight = (mass - full) * 1.0                                # weight minus buoyancy (g = 1)
    assert abs(float(k * v_t) - float(net_weight)) < 1e-6 * float(net_weight)   # drag = net weight at v_t


def test_contact_bounce_and_friction():
    normal, friction = eq.contact_velocity_changes(torch.tensor(2.0), torch.tensor(10.0))
    assert abs(float(normal) - 2.0 * (1 + eq.RESTITUTION)) < 1e-6   # bounces back with e = 0.3
    assert abs(float(friction) - eq.WALL_FRICTION * float(normal)) < 1e-6   # fast slide: Coulomb limit
    _, friction = eq.contact_velocity_changes(torch.tensor(2.0), torch.tensor(0.1))
    assert abs(float(friction) - 0.1 * 2 / 7) < 1e-6                # slow slide: at most stops the slip (2/7 rule)


# ---------------------------------------------------------------------------------------------- integration
def run_with_calm_water(density, start, seconds, tilt_deg=0.0, velocity=(0.0, 0.0, 0.0)):
    """Stone in calm water (the water is held still, no network)."""
    dt = (1 / 60) / SCALES.time                                     # one frame
    sim = Simulation(1, COARSE, "cpu", dt)                          # one scene
    index = torch.tensor([0])
    sim.reset(index, torch.tensor([0.25 / SCALES.length]), torch.tensor([math.radians(tilt_deg)]))   # calm water
    calm = type(sim.water)(sim.water.potential.clone(), sim.water.pressure.clone(), sim.water.phi.clone())
    sim.throw_stone(0, start, velocity, RADIUS, density)            # release the stone
    lowest_velocity, path = 0.0, []
    for _ in range(int(seconds / (dt * SCALES.time))):
        sim.advance(index, calm)                                    # water stays calm; tilt and stone move
        lowest_velocity = min(lowest_velocity, float(sim.stones.velocity[0, 1]))
        path.append(sim.stones.centre[0].clone())
    return sim, lowest_velocity, path


def test_heavy_stone_sinks_and_rests_on_the_floor():
    sim, lowest, _ = run_with_calm_water(2.5, (0.0, -0.16, 0.0), seconds=3.0)   # released just under the surface
    floor = -1.0 + RADIUS                                           # centre height when resting on the floor
    assert abs(float(sim.stones.centre[0, 1]) - floor) < 0.01       # on the floor
    assert sim.stones.velocity[0].norm() < 0.01                     # at rest
    terminal = eq.check_terminal_velocity(2.5, RADIUS)              # C3
    assert 0.5 * terminal < -lowest < 1.05 * terminal               # sinks at a believable speed, never faster than terminal


def fraction_under_water(sim):
    depth = float(sim.stones.surface_height[0] - sim.stones.centre[0, 1])     # centre below the surface
    return float(eq.underwater_volume(torch.tensor(RADIUS), torch.tensor(depth)) / eq.stone_volume(torch.tensor(RADIUS)))


def floating_depth(density):
    """Centre depth at which the fraction under water equals the density ratio (found by bisection)."""
    low, high = -RADIUS, RADIUS                                     # fully out ... fully under
    for _ in range(60):
        middle = 0.5 * (low + high)
        under = float(eq.underwater_volume(torch.tensor(RADIUS), torch.tensor(middle)) / eq.stone_volume(torch.tensor(RADIUS)))
        low, high = (middle, high) if under < density else (low, middle)
    return 0.5 * (low + high)


@pytest.mark.parametrize("density", [0.3, 0.5, 0.8])
def test_light_stone_placed_at_its_floating_depth_stays_there(density):
    surface = 0.0                                                   # 25 cm of water: the surface is at y = 0
    start = (0.0, surface - floating_depth(density), 0.0)           # exactly at the floating depth (C4)
    sim, _, path = run_with_calm_water(density, start, seconds=3.0)
    heights = torch.stack(path)[:, 1]
    assert (heights - start[1]).abs().max() < 0.005                 # stays put (less than 1.25 mm of drift)
    assert abs(fraction_under_water(sim) - density) < 0.01


@pytest.mark.parametrize("density", [0.3, 0.5, 0.8])
def test_light_stone_released_under_water_bobs_around_the_right_depth(density):
    # with the water held perfectly still there are no waves to carry energy away, so the stone keeps bobbing
    # (drag alone damps slowly); the CENTRE of the bobbing must be at the floating depth
    sim, _, path = run_with_calm_water(density, (0.0, -0.5, 0.0), seconds=8.0)
    heights = torch.stack(path)[-180:, 1]                           # the last 3 seconds
    middle = 0.5 * float(heights.max() + heights.min())             # centre of the bobbing
    expected = 0.0 - floating_depth(density)                        # floating height of the centre
    assert abs(middle - expected) < 0.03                            # within 7.5 mm
    assert float(heights.max() - heights.min()) < 0.5               # the bobbing stays small (< 12 cm peak to peak)


def test_heavy_stone_rolls_downhill_when_the_box_is_tilted():
    floor = -1.0 + RADIUS + 0.002
    sim, _, path = run_with_calm_water(2.5, (1.0, floor, 0.0), seconds=3.0, tilt_deg=10.0)   # right end raised
    assert float(path[-1][0]) < 1.0 - 0.5                           # moved left (downhill) by more than 12 cm
    sim, _, path = run_with_calm_water(2.5, (-1.0, floor, 0.0), seconds=3.0, tilt_deg=-10.0)  # left end raised
    assert float(path[-1][0]) > -1.0 + 0.5                          # moved right (downhill)


def test_stones_never_leave_the_box():
    for velocity in ((3.0, 0.0, 2.0), (-3.0, 0.0, -3.0), (0.0, 0.0, 3.0)):   # hard throws against the walls
        _, _, path = run_with_calm_water(2.5, (0.0, 0.8, 0.0), seconds=2.0, velocity=velocity)
        positions = torch.stack(path)                               # every position along the way
        half = torch.tensor([2.0, 1.0, 1.0])                        # half the inside size (dimensionless)
        assert torch.all(positions.abs() <= half - RADIUS + 1e-4)   # always inside


def test_stone_fields_on_the_grid(geometry):
    from scene import stone_fields                                  # the stone as the network sees it
    positions, _, h = geometry
    centre, radius = torch.tensor([[0.3, -0.2, 0.1]]), torch.tensor([0.2])
    mask, velocity = stone_fields(positions, centre, torch.tensor([[0.0, -1.0, 0.0]]), torch.tensor([[0.0, 0.0, 2.0]]),
                                  radius, torch.tensor([True]))
    volume = float(mask.sum()) * h ** 3                             # volume of the stone's cells
    assert abs(volume - 4 / 3 * math.pi * 0.008) < 0.15 * 4 / 3 * math.pi * 0.008   # close to the sphere's volume
    inside = mask[0, 0] > 0
    expected_vx = -2.0 * (positions[0, 1] - centre[0, 1])           # omega x r, x part: -w_z (y - y_s)
    assert torch.allclose(velocity[0, 0][inside], expected_vx[inside], atol=1e-5)   # rigid rotation
    assert torch.all(velocity[:, :, ~inside] == 0)                  # nothing outside the stone
