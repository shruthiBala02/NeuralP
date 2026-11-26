"""
Tests of model.py: shapes on both grids, the built-in constraints, the untrained starting point and gradients.
"""

import math                                                         # radians

import torch                                                        # tensors

from physics_neural_solver import physical_equations as eq          # equations
from physics_neural_solver.model import (INPUT_CHANNELS, OUTPUT_CHANNELS, BoxInputs, PhysicsNeuralSolver,
                                         WaterState, count_weights, network_input)
from scene import calm_water, water_volume                          # starting state and volume


def make_inputs(positions, solid, batch=2):
    depth = torch.full((batch,), 1.0)                               # 0.25 m
    tilt = torch.linspace(0.0, math.radians(10.0), batch)           # different tilts
    phi, potential, pressure = calm_water(positions, depth, tilt)
    state = WaterState(potential + 0.01 * torch.randn_like(potential), pressure, phi)   # a little motion
    box = BoxInputs(solid, torch.zeros_like(phi), torch.zeros_like(potential), tilt, torch.zeros(batch), torch.zeros(batch))
    return state, box


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


def test_untrained_network_predicts_no_change(geometry):
    positions, solid, h = geometry
    state, box = make_inputs(positions, solid)
    next_state = PhysicsNeuralSolver("small")(state, box, h)        # last layer starts at zero
    assert torch.allclose(next_state.phi, state.phi)                # surface unchanged
    assert torch.allclose(next_state.potential, state.potential)    # flow unchanged


def test_built_in_constraints_hold_for_a_random_network(geometry):
    positions, solid, h = geometry
    state, box = make_inputs(positions, solid)
    model = PhysicsNeuralSolver("small")
    torch.nn.init.normal_(model.network.output.weight, std=0.1)     # make the network output something non-zero
    next_state = model(state, box, h)
    assert torch.all(next_state.pressure[next_state.phi <= 0] == 0)  # eq. 8: p = 0 above the water
    change = next_state.potential - state.potential
    assert change.mean(dim=(2, 3, 4)).abs().max() < 1e-5            # A's mean is kept fixed
    u = model.velocity(next_state, h)
    divergence = eq.divergence_of_velocity(u, h)[:, :, 2:-2, 2:-2, 2:-2]
    assert divergence.abs().max() < 1e-4 * (u.abs().max() / h)      # eq. 1: incompressible, whatever the network says


def test_gradients_reach_every_weight(geometry):
    positions, solid, h = geometry
    state, box = make_inputs(positions, solid)
    model = PhysicsNeuralSolver("small")
    torch.nn.init.normal_(model.network.output.weight, std=0.01)    # (with a zero last layer, earlier layers get no gradient)
    next_state = model(state, box, h)
    volume = water_volume(state.phi, solid, box.stone_mask, h)
    terms = eq.loss_terms(state, next_state, box, positions, volume, h, dt=0.1)
    sum(terms.values()).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"no gradient for: {missing}"


def test_network_sizes():
    sizes = {name: count_weights(PhysicsNeuralSolver(name)) for name in ("small", "medium", "large")}
    assert sizes["small"] < sizes["medium"] < sizes["large"]        # bigger is bigger
    assert 1e6 < sizes["small"] < 2e6                               # ~1.4 million
    assert OUTPUT_CHANNELS == 5
