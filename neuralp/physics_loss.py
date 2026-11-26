"""
The physics loss: the residuals of the equations in PLAN.md, computed on the grid.

    loss terms (eq. numbers as in PLAN.md)
      2   momentum (Navier-Stokes), in water cells, on cell faces
      5   level-set transport, near the surface
      6   eikonal |grad phi| = 1, near the surface
      7   volume conservation, whole box
      9   free-surface no-shear, thin band just above the surface
      10  wall no-penetration, on the faces between water cells and wall cells
      11  wall law (with roughness, eq. 12), in the first water layer next to each wall
      13  water-stone boundary condition, inside the stone
    built in (not in the loss)
      1   mass conservation: u = curl(A)
      8   p = 0 above the surface: the pressure is multiplied by H(phi)

Time stepping: u' = (u_now + u_next) / 2 (IMEX). Everything is dimensionless.
NOTE: this is the first version. The advection term is computed at cell centres and moved to the
faces by averaging; a more careful (conservative) form can replace it later.
The loss weights are fixed here; the adaptive weighting comes with the training code.
"""

from dataclasses import dataclass

import torch

from . import grid
from .config import Scales, WaterConstants
from .scene import SceneBatch, smooth_heaviside

DEFAULT_LOSS_WEIGHTS = {
    "momentum": 1.0,
    "level_set": 1.0,
    "eikonal": 0.1,
    "volume": 1.0,
    "no_shear": 0.1,
    "no_penetration": 20.0,
    "wall_law": 1.0,
    "stone": 20.0,
}


@dataclass
class NextState:
    potential: torch.Tensor
    pressure: torch.Tensor
    phi: torch.Tensor


def advance_state(scene: SceneBatch, network_output: torch.Tensor) -> NextState:
    """Add the predicted change to the current state; apply the built-in constraints."""
    delta_potential = network_output[:, 0:3].float()
    delta_potential = delta_potential - delta_potential.mean(dim=(2, 3, 4), keepdim=True)  # A is defined up to a constant
    phi_next = scene.phi + network_output[:, 4:5].float()
    above_water_is_zero = smooth_heaviside(phi_next, 1.5 * scene.cell_size)
    pressure_next = (scene.pressure + network_output[:, 3:4].float()) * above_water_is_zero  # eq. 8
    return NextState(potential=scene.potential + delta_potential, pressure=pressure_next, phi=phi_next)


def masked_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.expand_as(values)
    return (weight * values).sum() / (weight.sum() + 1e-8)


def frame_forces(scene: SceneBatch, velocity_at_centres: torch.Tensor) -> torch.Tensor:
    """-dOmega/dt x r  -  Omega x (Omega x r)  -  2 Omega x u, with Omega = (0, 0, Omega_z) (eq. 2)."""
    batch_size = velocity_at_centres.shape[0]
    rate = scene.tilt_rate.view(batch_size, 1, 1, 1, 1)
    acceleration = scene.tilt_acceleration.view(batch_size, 1, 1, 1, 1)
    x, y = scene.cell_positions[:, 0:1], scene.cell_positions[:, 1:2]
    ux, uy = velocity_at_centres[:, 0:1], velocity_at_centres[:, 1:2]
    force_x = acceleration * y + rate ** 2 * x + 2.0 * rate * uy
    force_y = -acceleration * x + rate ** 2 * y - 2.0 * rate * ux
    return torch.cat([force_x, force_y, torch.zeros_like(force_x)], dim=1)


def friction_velocity(tangential_speed, distance, roughness, viscosity, water: WaterConstants, iterations=6):
    """Solve the wall law (eqs. 11-12) for u_tau by fixed-point iteration (no gradient through it)."""
    with torch.no_grad():
        kappa, offset, fit = water.von_karman_constant, water.log_law_offset, water.roughness_fit_constant
        u_tau = torch.sqrt(viscosity * tangential_speed / distance + 1e-12)
        for _ in range(iterations):
            y_plus = torch.clamp(distance * u_tau / viscosity, min=11.0)  # below ~11 the linear law applies
            roughness_plus = roughness * u_tau / viscosity
            shift = torch.log1p(fit * roughness_plus) / kappa
            u_tau = tangential_speed / (torch.log(y_plus) / kappa + offset - shift)
            u_tau = torch.clamp(u_tau, min=1e-8)
    return u_tau


def wall_law_residual(velocity_c, eddy_viscosity, scene: SceneBatch, viscosity, water: WaterConstants):
    """Ghost-cell wall function: the discrete wall stress must equal rho * u_tau^2 (dimensionless: u_tau^2)."""
    h = scene.cell_size
    batch_size = velocity_c.shape[0]
    roughness = scene.roughness.view(batch_size, 1, 1, 1, 1)
    squared_residuals = []
    for normal_axis, dim in enumerate((grid.X, grid.Y, grid.Z)):
        n = velocity_c.size(dim)
        tangential = [axis for axis in range(3) if axis != normal_axis]
        for fluid_index, wall_index in ((1, 0), (n - 2, n - 1)):
            u_fluid = velocity_c.narrow(dim, fluid_index, 1)[:, tangential]
            u_wall = velocity_c.narrow(dim, wall_index, 1)[:, tangential]
            nu_layer = viscosity + eddy_viscosity.narrow(dim, fluid_index, 1)
            speed = u_fluid.norm(dim=1, keepdim=True)
            u_tau = friction_velocity(speed, 0.5 * h, roughness, viscosity, water)
            model_stress = nu_layer * (u_fluid - u_wall) / h
            target_stress = u_tau ** 2 * u_fluid / (speed + 1e-8)
            squared_residuals.append(((model_stress - target_stress) ** 2).sum(dim=1).mean())
    return torch.stack(squared_residuals).mean()


def physics_loss(scene: SceneBatch, next_state: NextState, time_step: float,
                 scales: Scales = Scales(), weights: dict = DEFAULT_LOSS_WEIGHTS):
    water = scales.water
    h = scene.cell_size
    viscosity = scales.dimensionless_viscosity
    width = 1.5 * h
    batch_size = scene.phi.shape[0]

    # velocities (u = curl A, eq. 1 built in) and the IMEX average u'
    u_now = grid.velocity_from_potential(scene.potential, h)
    u_next = grid.velocity_from_potential(next_state.potential, h)
    u_mid = 0.5 * (u_now + u_next)
    u_mid_c = grid.faces_to_centres(u_mid)
    u_next_c = grid.faces_to_centres(u_next)

    fluid = (1.0 - scene.solid_mask) * (1.0 - scene.stone_mask)
    phi_mid = 0.5 * (scene.phi + next_state.phi)
    water_c = smooth_heaviside(phi_mid, width) * fluid
    water_faces = grid.scalar_centres_to_faces(water_c)

    # ---- eq. 2: momentum (Navier-Stokes) on the faces ----
    gradient = grid.velocity_gradient(u_mid_c, h)                     # (B, 3, 3, X, Y, Z)
    advection_c = torch.einsum("bj...,bij...->bi...", u_mid_c, gradient)
    strain = 0.5 * (gradient + gradient.transpose(1, 2))
    strain_magnitude = torch.sqrt(2.0 * (strain ** 2).sum(dim=(1, 2)) + 1e-12).unsqueeze(1)
    eddy_viscosity = (water.smagorinsky_constant * h) ** 2 * strain_magnitude  # eq. 4
    eddy_gradient = grid.central_gradient(eddy_viscosity, h)
    variable_viscosity_c = torch.einsum("bj...,bij...->bi...", eddy_gradient, 2.0 * strain)
    viscous = ((viscosity + grid.scalar_centres_to_faces(eddy_viscosity)) * grid.laplacian(u_mid, h)
               + grid.centres_to_faces(variable_viscosity_c))
    gravity = scene.gravity.view(batch_size, 3, 1, 1, 1)               # eq. 3, |g| = 1
    momentum_residual = ((u_next - u_now) / time_step
                         + grid.centres_to_faces(advection_c)
                         + grid.pressure_gradient_on_faces(next_state.pressure, h)
                         - viscous
                         - gravity
                         - grid.centres_to_faces(frame_forces(scene, u_mid_c)))
    loss_momentum = masked_mean(momentum_residual ** 2, water_faces)

    # ---- eq. 5: level-set transport, near the surface ----
    band = (phi_mid.abs() < 3.0 * h).float() * (1.0 - scene.solid_mask)
    level_set_residual = ((next_state.phi - scene.phi) / time_step
                          + grid.second_order_upwind_advection(phi_mid, u_mid_c, h))
    loss_level_set = masked_mean(level_set_residual ** 2, band)

    # ---- eq. 6: eikonal ----
    phi_gradient = grid.central_gradient(next_state.phi, h)
    gradient_norm = torch.sqrt((phi_gradient ** 2).sum(dim=1, keepdim=True) + 1e-12)
    band_next = (next_state.phi.abs() < 3.0 * h).float() * (1.0 - scene.solid_mask)
    loss_eikonal = masked_mean((gradient_norm - 1.0) ** 2, band_next)

    # ---- eq. 7: volume ----
    volume_next = (smooth_heaviside(next_state.phi, width) * fluid).sum(dim=(1, 2, 3, 4)) * h ** 3
    loss_volume = (((volume_next - scene.initial_volume) / scene.initial_volume) ** 2).mean()

    # ---- eq. 9: no shear just above the surface ----
    air_band = ((next_state.phi < 0) & (next_state.phi > -3.0 * h)).float() * (1.0 - scene.solid_mask)
    normal = phi_gradient / gradient_norm
    next_gradient = grid.velocity_gradient(u_next_c, h)
    normal_derivative = torch.einsum("bj...,bij...->bi...", normal, next_gradient)
    loss_no_shear = masked_mean((normal_derivative ** 2).sum(dim=1, keepdim=True), air_band)

    # ---- eq. 10: no flow through the walls (normal velocity on water/wall faces) ----
    wall_normal_velocities = []
    for axis, dim in enumerate((grid.X, grid.Y, grid.Z)):
        component = u_next[:, axis:axis + 1]
        n = component.size(dim)
        wall_normal_velocities += [component.narrow(dim, 0, 1), component.narrow(dim, n - 2, 1)]
    loss_no_penetration = torch.stack([(v ** 2).mean() for v in wall_normal_velocities]).mean()

    # ---- eqs. 11-12: wall law with roughness ----
    loss_wall_law = wall_law_residual(u_next_c, eddy_viscosity, scene, viscosity, water)

    # ---- eq. 13: water moves with the stone ----
    stone_weight = scene.stone_mask * (1.0 - scene.solid_mask)
    loss_stone = masked_mean(((u_next_c - scene.stone_velocity_field) ** 2).sum(dim=1, keepdim=True), stone_weight)

    terms = {
        "momentum": loss_momentum,
        "level_set": loss_level_set,
        "eikonal": loss_eikonal,
        "volume": loss_volume,
        "no_shear": loss_no_shear,
        "no_penetration": loss_no_penetration,
        "wall_law": loss_wall_law,
        "stone": loss_stone,
    }
    total = sum(weights[name] * value for name, value in terms.items())
    return total, terms
