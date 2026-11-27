"""
Tests of symmetry.py: mirroring a scene front-to-back must give a scene with exactly the same physics.
If any staggering or sign were wrong, the mirrored scene would break the equations differently - these tests catch it.
"""

import math                                                         # radians

import torch                                                        # tensors

from physics_neural_solver import physical_equations as eq          # the equations
from physics_neural_solver.model import BoxInputs, PhysicsNeuralSolver, WaterState
from physics_neural_solver.symmetry import (mirror_box, mirror_potential, mirror_some, mirror_state,
                                            mirror_velocity_faces)
from scene import calm_water, stone_fields, water_volume            # starting state, stone, volume


def moving_scene(positions, solid, h):
    """Tilted water with random flow, a turning box and a moving, spinning stone placed off-centre in z."""
    torch.manual_seed(3)
    tilt = torch.tensor([math.radians(6.0)])
    phi, potential, pressure = calm_water(positions, torch.tensor([1.0]), tilt)
    shape = positions.shape[2:]
    potential = 0.03 * torch.randn(1, 3, *shape) * eq.wall_potential_mask(shape, positions.device)   # some flow
    centre = torch.tensor([[0.3, -0.4, 0.35]])                      # off-centre in z, so mirroring moves it
    mask, field = stone_fields(positions, centre, torch.tensor([[0.2, -0.5, 0.4]]), torch.tensor([[1.0, -2.0, 3.0]]),
                               torch.tensor([0.16]), torch.tensor([True]))
    volume = water_volume(phi, solid, torch.zeros_like(phi), h)
    box = BoxInputs(solid, mask, field, tilt, torch.tensor([0.05]), torch.tensor([-0.3]), initial_volume=volume)
    return WaterState(potential, pressure, phi), box


def test_mirroring_twice_gives_the_original(geometry):
    positions, solid, h = geometry
    state, box = moving_scene(positions, solid, h)
    twice = mirror_state(mirror_state(state))
    assert torch.equal(twice.potential, state.potential) and torch.equal(twice.phi, state.phi)
    box2 = mirror_box(mirror_box(box))
    assert torch.equal(box2.stone_velocity_field, box.stone_velocity_field)


def test_velocity_of_the_mirrored_flow_is_the_mirrored_velocity(geometry):
    positions, solid, h = geometry
    state, _ = moving_scene(positions, solid, h)
    u_then_mirror = mirror_velocity_faces(eq.velocity_from_vector_potential(state.potential, h))
    mirror_then_u = eq.velocity_from_vector_potential(mirror_potential(state.potential), h)
    assert torch.allclose(u_then_mirror, mirror_then_u, atol=1e-5)   # everywhere, wall cells included


def test_mirrored_scene_has_exactly_the_same_physics_loss(geometry):
    positions, solid, h = geometry
    state, box = moving_scene(positions, solid, h)
    model = PhysicsNeuralSolver("small")
    torch.nn.init.normal_(model.network.output.weight, std=0.02)    # some prediction (not symmetric itself)
    with torch.no_grad():
        next_state = model(state, box, h)
        original = eq.loss_terms(state, next_state, box, positions, box.initial_volume, h, dt=0.1)
        mirrored = eq.loss_terms(mirror_state(state), mirror_state(next_state), mirror_box(box), positions,
                                 box.initial_volume, h, dt=0.1)
    for name in original:
        a, b = float(original[name]), float(mirrored[name])
        assert abs(a - b) <= 1e-4 * max(abs(a), 1e-8) + 1e-10, f"{name}: {a} vs {b}"   # the same physics


def test_mirror_some_only_touches_the_chosen_scenes(geometry):
    positions, solid, h = geometry
    state, box = moving_scene(positions, solid, h)
    two = WaterState(state.potential.repeat(2, 1, 1, 1, 1), state.pressure.repeat(2, 1, 1, 1, 1), state.phi.repeat(2, 1, 1, 1, 1))
    box2 = BoxInputs(solid, box.stone_mask.repeat(2, 1, 1, 1, 1), box.stone_velocity_field.repeat(2, 1, 1, 1, 1),
                     box.tilt.repeat(2), box.tilt_rate.repeat(2), box.tilt_acceleration.repeat(2), box.initial_volume.repeat(2))
    out, _ = mirror_some(two, box2, torch.tensor([False, True]))
    assert torch.equal(out.potential[0], state.potential[0])        # scene 0 untouched
    assert torch.equal(out.potential[1], mirror_state(state).potential[0])   # scene 1 mirrored
