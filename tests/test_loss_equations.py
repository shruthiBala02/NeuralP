"""
Tests of Sections 3 and 4 (loss equations and starting conditions): states with KNOWN answers must give the
expected residuals. The most important test: calm water is an exact solution, so every loss term must be ~0.
"""

import math                                                         # radians

import pytest                                                       # parametrised tests
import torch                                                        # tensors

from config import Scales                                           # reference scales
from physics_neural_solver import physical_equations as eq          # the equations under test
from physics_neural_solver.model import BoxInputs, WaterState       # containers
from scene import calm_water, water_volume                          # starting state and volume

INNER = (slice(None), slice(None), slice(2, -2), slice(2, -2), slice(2, -2))   # cells away from the edges


def calm_scene(positions, solid, depth_m, tilt_deg):
    """Calm water state and box inputs for one scene."""
    scales = Scales()
    depth = torch.tensor([depth_m / scales.length])                 # dimensionless depth
    tilt = torch.tensor([math.radians(tilt_deg)])                   # tilt angle
    phi, potential, pressure = calm_water(positions, depth, tilt)   # eqs. 14-16
    state = WaterState(potential=potential, pressure=eq.apply_free_surface_pressure(pressure, phi, 0.1), phi=phi)
    box = BoxInputs(solid_mask=solid, stone_mask=torch.zeros_like(phi), stone_velocity_field=torch.zeros_like(potential),
                    tilt=tilt, tilt_rate=torch.zeros(1), tilt_acceleration=torch.zeros(1))
    return state, box


# ---------------------------------------------------------------------------------------------- starting state
@pytest.mark.parametrize("depth_m, tilt_deg", [(0.10, 0.0), (0.25, 0.0), (0.35, 0.0), (0.25, 10.0), (0.20, -15.0)])
def test_starting_state(geometry, depth_m, tilt_deg):
    positions, solid, h = geometry
    state, box = calm_scene(positions, solid, depth_m, tilt_deg)
    phi_gradient = eq.gradient_at_centres(state.phi, h)[INNER]      # phi must be a true distance: |grad phi| = 1
    assert torch.allclose(phi_gradient.norm(dim=1), torch.ones_like(phi_gradient[:, 0]), atol=1e-4)
    normal = -eq.gravity_in_box_frame(box.tilt).view(3)             # "up"
    assert torch.allclose(phi_gradient.mean(dim=(2, 3, 4)).view(3), -normal, atol=1e-4)   # surface at right angles to gravity
    water = state.phi > 0                                           # hydrostatic pressure = depth below the surface
    assert torch.allclose(state.pressure[water], state.phi[water], atol=1e-6)
    assert torch.all(state.pressure[~water] == 0)                   # 0 above the water
    assert torch.all(state.potential == 0)                          # at rest
    if tilt_deg == 0.0:                                             # level box: volume = L x W x depth
        volume = water_volume(state.phi, solid, torch.zeros_like(state.phi), h) * Scales().length ** 3   # m^3
        assert abs(float(volume) - 1.0 * 0.5 * depth_m) < 0.01 * 0.5 * depth_m   # within 1 %


# ---------------------------------------------------------------------------------------------- calm water = exact solution
@pytest.mark.parametrize("depth_m, tilt_deg", [(0.25, 0.0), (0.13, 0.0), (0.31, 7.0), (0.25, -12.0)])
def test_calm_water_satisfies_every_loss_equation(geometry, depth_m, tilt_deg):
    positions, solid, h = geometry
    state, box = calm_scene(positions, solid, depth_m, tilt_deg)
    volume = water_volume(state.phi, solid, box.stone_mask, h)      # V_0
    terms = eq.loss_terms(state, state, box, positions, volume, h, dt=0.1)   # "next" = "now": the water does not move
    for name, value in terms.items():
        assert float(value) < 1e-4, f"{name} = {float(value)} for calm water (should be ~0)"


def test_water_at_rest_without_pressure_fails_momentum(geometry):
    positions, solid, h = geometry
    state, box = calm_scene(positions, solid, 0.25, 0.0)
    no_pressure = WaterState(state.potential, torch.zeros_like(state.pressure), state.phi)   # gravity not balanced
    volume = water_volume(state.phi, solid, box.stone_mask, h)
    terms = eq.loss_terms(state, no_pressure, box, positions, volume, h, dt=0.1)
    assert float(terms["momentum (eq. 2)"]) > 0.3                   # the vertical residual is g = 1 (mean over 3 components ~1/3): noticed


# ---------------------------------------------------------------------------------------------- momentum: free fall
def test_momentum_free_fall_has_zero_residual(geometry):
    positions, solid, h = geometry
    dt = 0.1
    edge_x = positions[:, 0:1] + 0.5 * h                            # x of the edges where A_z lives
    potential_next = torch.zeros(1, 3, *positions.shape[2:])
    potential_next[:, 2:3] = dt * edge_x                            # u_y = -dA_z/dx = -dt: fallen for one step under g = 1
    u_now = torch.zeros(1, 3, *positions.shape[2:])                 # at rest before
    u_next = eq.velocity_from_vector_potential(potential_next, h)   # uniform downward velocity after
    phi = torch.full_like(positions[:, 0:1], 10.0)                  # everything is water (no surface nearby)
    faces = torch.ones(1, 3, *positions.shape[2:])                  # check every face
    residual, _, _ = eq.momentum_residual(u_now, u_next, torch.zeros_like(phi), phi, faces, positions,
                                          eq.gravity_in_box_frame(torch.zeros(1)), torch.zeros(1), torch.zeros(1),
                                          h, dt, 1e-6, Scales().water)
    assert residual[INNER].abs().max() < 1e-3                       # du/dt = g exactly: nothing else acts


# ---------------------------------------------------------------------------------------------- level set and eikonal
def test_level_set_moving_with_the_water_has_zero_residual(geometry):
    positions, solid, h = geometry
    dt, rise = 0.1, 0.3                                             # the surface rises at 0.3
    phi_now = 0.05 - positions[:, 1:2]                              # flat surface
    phi_next = phi_now + rise * dt                                  # moved up by rise * dt
    velocity = torch.zeros(1, 3, *positions.shape[2:])
    velocity[:, 1] = rise                                           # the water moves up at the same speed
    residual, weight = eq.level_set_residual(phi_now, phi_next, velocity, solid, h, dt)
    assert (residual[INNER] * weight[INNER]).abs().max() < 1e-4     # d(phi)/dt + u.grad(phi) = 0
    wrong = phi_now + 2 * rise * dt                                 # surface moving twice as fast as the water
    residual, weight = eq.level_set_residual(phi_now, wrong, velocity, solid, h, dt)
    assert float(eq.masked_mean(residual ** 2, weight)) > 0.01      # the loss notices


def test_eikonal_detects_a_stretched_level_set(geometry):
    positions, solid, h = geometry
    good = 0.05 - positions[:, 1:2]                                 # true distance
    residual, weight = eq.eikonal_residual(good, solid, h)
    assert float(eq.masked_mean(residual ** 2, weight)) < 1e-8      # |grad phi| = 1
    residual, weight = eq.eikonal_residual(2.0 * good, solid, h)    # stretched: |grad phi| = 2
    assert abs(float(eq.masked_mean(residual, weight)) - 1.0) < 1e-3


# ---------------------------------------------------------------------------------------------- volume
def test_volume_change_is_measured_correctly(geometry):
    positions, solid, h = geometry
    phi = 0.0 - positions[:, 1:2]                                   # surface at y = 0 (depth 0.25 m)
    fluid = 1.0 - solid
    volume = water_volume(phi, solid, torch.zeros_like(phi), h)     # V_0
    raised = phi + 0.04                                             # surface 0.04 (1 cm) higher
    relative, _ = eq.volume_residual(raised, fluid, volume, h)      # eq. 7
    expected = 0.04 * 4.0 * 2.0 / float(volume)                     # extra volume = rise x length x width (dimensionless)
    assert abs(float(relative) - expected) < 0.02 * expected


# ---------------------------------------------------------------------------------------------- walls and stone
def test_no_penetration(geometry):
    positions, _, h = geometry
    still = torch.zeros(1, 3, *positions.shape[2:])
    residual, _ = eq.no_penetration_residual(still)
    assert residual.abs().max() == 0                                # no flow: nothing goes through the walls
    into_wall = still.clone()
    into_wall[:, 0] = 0.5                                           # flow along x everywhere
    residual, _ = eq.no_penetration_residual(into_wall)
    assert residual.abs().max() > 0.49                              # it goes through the x-walls: detected


def test_wall_law_is_zero_without_flow(geometry):
    positions, _, h = geometry
    still = torch.zeros(1, 3, *positions.shape[2:])
    residual, _ = eq.wall_law_residual(still, torch.zeros_like(still[:, :1]), h, 2.6e-6, Scales().water)
    assert residual.abs().max() < 1e-12                             # no flow, no friction


def test_stone_boundary(geometry):
    positions, solid, h = geometry
    stone_mask = ((positions - torch.tensor([0.0, -0.3, 0.0]).view(1, 3, 1, 1, 1)).norm(dim=1, keepdim=True) < 0.2).float()
    field = torch.zeros(1, 3, *positions.shape[2:])
    field[:, 1:2] = -0.5 * stone_mask                               # the stone moves down at 0.5
    water_with_stone = torch.zeros_like(field)
    water_with_stone[:, 1] = -0.5                                   # water everywhere moving with it (on every face)
    residual, weight = eq.stone_boundary_residual(water_with_stone, field, stone_mask, solid)
    assert float(eq.masked_mean(residual ** 2, weight)) < 1e-12     # water moving with the stone: 0
    residual, weight = eq.stone_boundary_residual(torch.zeros_like(field), field, stone_mask, solid)
    assert float(eq.masked_mean(residual ** 2, weight)) > 0.05      # water ignoring the stone (flows through it): detected
    assert float(weight[:, 1:2][..., :].sum()) > float(stone_mask.sum())   # boundary faces are checked, not only the inside


def test_turning_deep_water_into_air_is_caught_by_the_surface_equation(geometry):
    positions, solid, h = geometry
    phi_now = 0.0 - positions[:, 1:2]                               # surface at y = 0 (25 cm of water)
    phi_next = phi_now.clone()
    deep = (positions[:, 1:2] < -0.6) & (positions[:, 0:1].abs() < 0.3)   # a patch of deep water ...
    phi_next[deep] = -0.5                                           # ... suddenly marked as air (a cheat)
    still = torch.zeros(1, 3, *positions.shape[2:])                 # no flow that could explain it
    residual, weight = eq.level_set_residual(phi_now, phi_next, still, solid, h, 0.1)
    assert float(eq.masked_mean(residual ** 2, weight)) > 1.0       # the surface equation notices


def test_calm_water_with_a_resting_stone_satisfies_every_equation(geometry):
    positions, solid, h = geometry
    state, box = calm_scene(positions, solid, 0.25, 0.0)
    centre = torch.tensor([[0.5, -1.0 + 0.16, 0.0]])                # a stone resting on the floor, under water
    from scene import stone_fields
    stone_mask, stone_velocity = stone_fields(positions, centre, torch.zeros(1, 3), torch.zeros(1, 3),
                                              torch.tensor([0.16]), torch.tensor([True]))
    box = BoxInputs(solid, stone_mask, stone_velocity, box.tilt, box.tilt_rate, box.tilt_acceleration)
    volume = water_volume(state.phi, solid, stone_mask, h)
    terms = eq.loss_terms(state, state, box, positions, volume, h, dt=0.1)
    for name, value in terms.items():
        assert float(value) < 1e-4, f"{name} = {float(value)} with a resting stone (should be ~0)"


def test_water_turning_with_the_box_needs_the_centrifugal_force(geometry):
    positions, solid, h = geometry
    rate = 1.0                                                      # a fast, steady rotation (to make the effect clear)
    x, y = positions[:, 0:1], positions[:, 1:2]
    gravity = eq.gravity_in_box_frame(torch.zeros(1))               # level box
    pressure_field = -y + 0.5 * rate ** 2 * (x ** 2 + y ** 2)       # hydrostatic in the rotating frame: grad p = g + w^2 r
    pressure_field = pressure_field - pressure_field.min() + 0.2    # positive everywhere (the box full of water)
    phi = torch.full_like(pressure_field, 10.0)                     # no free surface nearby
    u = torch.zeros(1, 3, *positions.shape[2:])                     # at rest relative to the box
    faces = torch.ones_like(u)
    inner = (slice(None), slice(None), slice(2, -2), slice(2, -2), slice(2, -2))

    def momentum(with_rotation):
        residual, _, _ = eq.momentum_residual(u, u, pressure_field, phi, faces, positions, gravity,
                                              torch.tensor([rate if with_rotation else 0.0]), torch.zeros(1),
                                              h, 0.1, 1e-6, Scales().water)
        return float((residual[inner] ** 2).mean())

    assert momentum(with_rotation=True) < 1e-3 * rate ** 4          # balanced: gravity + centrifugal = pressure gradient
    assert momentum(with_rotation=False) > 0.1 * rate ** 4          # without the centrifugal force: not balanced
