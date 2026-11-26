"""
Tests of model.py: shapes on both grids, the STRICT RULES (they must hold whatever the network outputs),
the untrained starting point and gradients.
"""

import math                                                         # radians

import torch                                                        # tensors

from physics_neural_solver import physical_equations as eq          # equations
from physics_neural_solver.model import (INPUT_CHANNELS, OUTPUT_CHANNELS, BoxInputs, PhysicsNeuralSolver,
                                         WaterState, count_weights, network_input)
from scene import calm_water, water_volume                          # starting state and volume


def make_inputs(positions, solid, batch=2, moving=True):
    h = float(positions[0, 0, 1, 0, 0] - positions[0, 0, 0, 0, 0])  # cell size (dimensionless)
    depth = torch.full((batch,), 1.0)                               # 0.25 m
    tilt = torch.linspace(0.0, math.radians(10.0), batch)           # different tilts
    phi, potential, pressure = calm_water(positions, depth, tilt)
    if moving:                                                      # a little (random) motion
        potential = (potential + 0.01 * torch.randn_like(potential)) * eq.wall_potential_mask(phi.shape[2:], phi.device)
    volume = water_volume(phi, solid, torch.zeros_like(phi), h)     # the water volume to keep (eq. 7)
    state = WaterState(potential, pressure, phi)
    box = BoxInputs(solid, torch.zeros_like(phi), torch.zeros_like(potential), tilt, torch.zeros(batch), torch.zeros(batch),
                    initial_volume=volume)
    return state, box


def random_network():
    model = PhysicsNeuralSolver("small")
    torch.nn.init.normal_(model.network.output.weight, std=0.1)     # make the network output something non-zero
    return model


def test_network_input_has_15_channels_in_the_right_order(geometry):
    positions, solid, _ = geometry
    state, box = make_inputs(positions, solid)
    x = network_input(state, box)
    assert x.shape[1] == INPUT_CHANNELS == 15
    assert torch.equal(x[:, 0:1], state.phi)                        # channel 0 = phi
    assert torch.equal(x[:, 1:4], state.potential)                  # channels 1-3 = A
    assert torch.equal(x[:, 5:6], solid.expand(2, -1, -1, -1, -1))  # channel 5 = walls
    assert torch.allclose(x[:, 10:13, 0, 0, 0], eq.gravity_in_box_frame(box.tilt))   # channels 10-12 = gravity


def test_output_shapes_on_both_grids(geometry):
    positions, solid, h = geometry
    state, box = make_inputs(positions, solid)
    next_state = PhysicsNeuralSolver("small")(state, box, h)
    assert next_state.potential.shape == state.potential.shape      # same grid in, same grid out (padding removed)
    assert next_state.pressure.shape == state.pressure.shape
    assert next_state.phi.shape == state.phi.shape


def test_untrained_network_keeps_calm_water_exactly(geometry):
    positions, solid, h = geometry
    state, box = make_inputs(positions, solid, moving=False)        # water at rest
    next_state = PhysicsNeuralSolver("small")(state, box, h)        # last layer starts at zero: no change
    assert torch.allclose(next_state.phi, state.phi, atol=1e-5)     # no flow: the surface stays where it is
    assert torch.allclose(next_state.potential, state.potential)    # flow unchanged
    assert torch.allclose(next_state.pressure, state.pressure)      # pressure unchanged


# ---------------------------------------------------------------------------------------------- the strict rules
def test_strict_rules_hold_for_a_random_network(geometry):
    positions, solid, h = geometry
    state, box = make_inputs(positions, solid)
    next_state = random_network()(state, box, h)                    # whatever the network outputs ...
    u = eq.velocity_from_vector_potential(next_state.potential, h)
    assert torch.all(next_state.pressure[next_state.phi <= 0] == 0)  # eq. 8: p = 0 above the water
    divergence = eq.divergence_of_velocity(u, h)[:, :, 2:-2, 2:-2, 2:-2]
    assert divergence.abs().max() < 1e-4 * (u.abs().max() / h)      # eq. 1: incompressible
    through_walls, _ = eq.no_penetration_residual(u)
    assert through_walls.abs().max() < 1e-6 * max(float(u.abs().max()), 1.0)   # eq. 10: nothing flows through any wall
    volume = water_volume(next_state.phi, solid, box.stone_mask, h)
    assert ((volume - box.initial_volume).abs() / box.initial_volume).max() < 1e-5   # eq. 7: exactly the same water


def test_no_flow_through_walls_also_for_huge_random_potentials(geometry):
    positions, solid, h = geometry
    shape = positions.shape[2:]
    potential = 5.0 * torch.randn(1, 3, *shape) * eq.wall_potential_mask(shape, positions.device)   # any A with the wall rule
    u = eq.velocity_from_vector_potential(potential, h)
    through_walls, _ = eq.no_penetration_residual(u)
    assert through_walls.abs().max() < 1e-4                         # exactly zero (up to rounding)
    along = u[:, 1:2, 1, 2:-2, 2:-2]                                 # but the flow ALONG the walls is free
    assert along.abs().max() > 1.0


def test_the_surface_is_carried_by_the_flow(geometry):
    positions, solid, h = geometry
    phi = 0.3 - positions[:, 0:1]                                   # a surface standing across x (water on the left)
    velocity = torch.zeros(1, 3, *positions.shape[2:])
    velocity[:, 0] = 0.5                                            # everything moves right at 0.5
    carried = eq.carry_surface(phi, velocity, h, dt=0.1)
    inner = (slice(None), slice(None), slice(3, -3), slice(1, -1), slice(1, -1))   # away from the walls
    assert torch.allclose(carried[inner], (phi + 0.5 * 0.1)[inner], atol=1e-5)   # moved by exactly u dt
    assert eq.carry_surface(phi, torch.zeros_like(velocity), h, dt=0.1).sub(phi)[inner].abs().max() < 1e-6   # no flow: no change


def test_volume_is_restored_exactly(geometry):
    positions, solid, h = geometry
    phi = 0.0 - positions[:, 1:2]                                   # surface at y = 0
    fluid = 1.0 - solid
    target = water_volume(phi, solid, torch.zeros_like(phi), h)
    shrunk = phi - 0.07                                             # 7 % of the depth lost (like a leaky carry)
    restored = eq.restore_volume(shrunk, fluid, target, h)
    assert abs(float(water_volume(restored, solid, torch.zeros_like(phi), h) / target) - 1.0) < 1e-5
    assert torch.allclose(restored, phi, atol=1e-4)                 # by moving the surface back up


def test_redistancing_makes_phi_a_distance_without_moving_the_surface(geometry):
    positions, solid, h = geometry
    y = positions[:, 1:2]
    stretched = 2.0 * (0.05 - y)                                    # right surface (y = 0.05), but |grad phi| = 2
    fixed = eq.redistance(stretched, solid, h, iterations=40)       # 40 pseudo-steps reach ~20 cells from the surface
    near = ((0.05 - y).abs() < 10 * h) & (solid == 0)               # check up to 10 cells from the surface
    near[:, :, :2] = near[:, :, -2:] = False                        # (not right next to the walls: they copy the
    near[:, :, :, :2] = near[:, :, :, -2:] = False                  #  inside, so the difference across them is halved)
    near[:, :, :, :, :2] = near[:, :, :, :, -2:] = False
    gradient = eq.gradient_at_centres(fixed, h).norm(dim=1, keepdim=True)
    assert (gradient[near] - 1.0).abs().max() < 0.05                # a true distance again
    assert ((fixed - (0.05 - y))[near]).abs().max() < 0.1 * h       # and the surface did not move


# ---------------------------------------------------------------------------------------------- energy rule, worst faces
def test_energy_rule(geometry):
    positions, solid, h = geometry
    state, box = make_inputs(positions, solid, batch=1, moving=False)   # water at rest, level box
    fluid = 1.0 - solid
    same, weight = eq.energy_rule_residual(state, state, box, positions, fluid, box.initial_volume, h, 0.1)
    assert float(same.abs().max()) == 0 and float(weight.sum()) == 1   # nothing changes: no residual (and it is checked)
    shape = positions.shape[2:]
    faster = WaterState(0.2 * torch.randn(1, 3, *shape) * eq.wall_potential_mask(shape, positions.device), state.pressure, state.phi)
    grows, _ = eq.energy_rule_residual(state, faster, box, positions, fluid, box.initial_volume, h, 0.1)
    assert float(grows) > 0                                         # water speeding up by itself: caught
    calms, _ = eq.energy_rule_residual(faster, state, box, positions, fluid, box.initial_volume, h, 0.1)
    assert float(calms) == 0                                        # water calming down is allowed (friction)
    box.tilt_rate = torch.tensor([0.1])                             # the box is turning: it may add energy
    _, weight = eq.energy_rule_residual(state, faster, box, positions, fluid, box.initial_volume, h, 0.1)
    assert float(weight.sum()) == 0                                 # so the rule is not applied


def test_worst_faces_are_not_hidden_by_the_average():
    spread = torch.full((1, 1, 100, 10, 10), 0.01)                  # small errors everywhere
    piled = torch.zeros_like(spread)
    piled.view(-1)[:100] = 1.0                                      # the same total, piled into 1 % of the faces
    weight = torch.ones_like(spread)
    assert abs(float(eq.masked_mean(spread, weight)) - float(eq.masked_mean(piled, weight))) < 1e-6   # same average ...
    assert float(eq.worst_mean(piled, weight, 0.01)) > 50 * float(eq.worst_mean(spread, weight, 0.01))   # ... but the worst differ


def test_gradients_reach_every_weight(geometry):
    positions, solid, h = geometry
    state, box = make_inputs(positions, solid)
    model = PhysicsNeuralSolver("small")
    torch.nn.init.normal_(model.network.output.weight, std=0.01)    # (with a zero last layer, earlier layers get no gradient)
    next_state = model(state, box, h)
    terms = eq.loss_terms(state, next_state, box, positions, box.initial_volume, h, dt=0.1)
    sum(terms.values()).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"no gradient for: {missing}"


def test_network_sizes():
    sizes = {name: count_weights(PhysicsNeuralSolver(name)) for name in ("small", "medium", "large")}
    assert sizes["small"] < sizes["medium"] < sizes["large"]        # bigger is bigger
    assert 1e6 < sizes["small"] < 2e6                               # ~1.4 million
    assert OUTPUT_CHANNELS == 4                                     # change of A (3) and p (1): phi is carried, not predicted
