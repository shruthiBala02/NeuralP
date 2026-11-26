"""
Tests of Sections 1 and 2 (built-in constraints and formulas): each is checked on a case with a known answer.
"""

import math                                                         # pi, radians

import torch                                                        # tensors

from config import WaterConstants                                   # constants
from physics_neural_solver import physical_equations as eq          # the formulas under test

INNER = (slice(None), slice(None), slice(2, -2), slice(2, -2), slice(2, -2))   # cells away from the edges


def test_free_surface_pressure_is_zero_above_the_water(geometry):
    positions, _, h = geometry
    phi = 0.1 - positions[:, 1:2]                                   # surface at y = 0.1
    pressure = torch.rand_like(phi)                                 # any pressure
    result = eq.apply_free_surface_pressure(pressure, phi, h)       # eq. 8
    assert torch.all(result[phi <= 0] == 0)                         # 0 above the water
    assert torch.allclose(result[phi > 0], pressure[phi > 0])       # untouched in the water


def test_smoothed_step_values():
    h = 0.1
    assert abs(float(eq.smoothed_step(torch.tensor(0.0), h)) - 0.5) < 1e-6   # half on the surface
    assert float(eq.smoothed_step(torch.tensor(1.0), h)) > 0.999    # deep in the water: 1
    assert float(eq.smoothed_step(torch.tensor(-1.0), h)) < 0.001   # well above the water: 0


def test_gravity_in_the_tilted_box():
    tilt = torch.tensor([0.0, math.radians(90.0), math.radians(-10.0)])
    g = eq.gravity_in_box_frame(tilt)                               # eq. 3
    assert torch.allclose(g[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)   # level: straight down
    assert torch.allclose(g[1], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)   # 90 degrees: along -x
    assert g[2, 0] > 0                                              # right end lowered: water flows to +x
    assert torch.allclose(g.norm(dim=1), torch.ones(3))             # always |g| = 1


def test_tilt_forces_centrifugal_coriolis_euler():
    position = torch.tensor([0.5, 0.2, 0.0]).view(1, 3, 1, 1, 1)    # a point in the box
    still = torch.zeros(1, 3, 1, 1, 1)                              # water at rest
    f = eq.tilt_forces(position, still, torch.tensor([2.0]), torch.tensor([0.0])).flatten()
    assert torch.allclose(f, torch.tensor([4.0 * 0.5, 4.0 * 0.2, 0.0]), atol=1e-6)   # centrifugal = w^2 r, outwards
    moving = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1, 1)      # water moving along +x
    f = eq.tilt_forces(torch.zeros(1, 3, 1, 1, 1), moving, torch.tensor([1.0]), torch.tensor([0.0])).flatten()
    assert torch.allclose(f, torch.tensor([0.0, -2.0, 0.0]), atol=1e-6)   # Coriolis = -2 Omega x u
    f = eq.tilt_forces(torch.tensor([0.0, 0.3, 0.0]).view(1, 3, 1, 1, 1), still, torch.tensor([0.0]), torch.tensor([1.0])).flatten()
    assert torch.allclose(f, torch.tensor([0.3, 0.0, 0.0]), atol=1e-6)    # Euler = -dOmega/dt x r


def test_smagorinsky_for_a_simple_shear_flow(geometry):
    positions, _, h = geometry
    shear = 0.8                                                     # u_x = 0.8 y
    velocity = torch.zeros(1, 3, *positions.shape[2:])
    velocity[:, 0] = shear * positions[:, 1]
    gradient = eq.velocity_gradient_at_centres(velocity, h)         # du_i/dx_j
    nu_t = eq.smagorinsky_eddy_viscosity(gradient, h, 0.1)[INNER]   # eq. 4
    expected = (0.1 * h) ** 2 * shear                               # |S| = shear rate for a simple shear
    assert torch.allclose(nu_t, torch.full_like(nu_t, expected), rtol=1e-3)


def test_ghost_fluid_gradient_balances_gravity_for_any_surface_position(geometry):
    positions, _, h = geometry
    for surface in (0.013, 0.05, 0.091, 0.2):                       # surfaces between cell centres, at different fractions
        phi = surface - positions[:, 1:2]                           # flat surface at y = surface
        pressure = torch.clamp(phi, min=0.0)                        # hydrostatic pressure, 0 above
        gradient, _ = eq.pressure_gradient_on_faces(pressure, phi, h)   # with the exact surface position
        wet0, wet1 = phi > 0, eq.next_cell(phi, eq.Y) > 0           # cells below and above each y-face
        touching_water = (wet0 | wet1)[INNER]                       # y-faces next to water (including the surface face)
        dp_dy = gradient[:, 1:2][INNER][touching_water]             # vertical pressure gradient there
        assert torch.allclose(dp_dy, torch.full_like(dp_dy, -1.0), atol=1e-4)   # = -rho g everywhere: exact balance


def test_wall_law_friction_velocity_solves_the_log_law():
    water = WaterConstants()
    viscosity, distance = 2.6e-6, 0.05                              # dimensionless nu and y = h/2
    for speed in (0.05, 0.3, 1.0):                                  # different flow speeds
        u_tau = eq.friction_velocity(torch.tensor(speed), torch.tensor(distance), viscosity, water)
        y_plus = distance * u_tau / viscosity                       # wall distance in wall units
        assert y_plus > 11                                          # in the log-law region
        law = torch.log(y_plus) / water.von_karman_constant + water.log_law_offset
        assert abs(float(speed / u_tau - law)) < 1e-3               # u_t / u_tau = (1/kappa) ln(y+) + B
