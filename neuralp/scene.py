"""
Random water states in the tilting box, used to fill the training pool
(and, for now, to feed the timing benchmark).

Each state is: calm water at a random depth with its surface at right angles to gravity,
a hydrostatic pressure (eqs. 14-16), a random tilt angle / tilt rate, a random wall
roughness preset and one sphere stone at a random place with a random velocity.
All values are dimensionless (see config.Scales). The grid is attached to the box and its
origin is the tilt pivot (the box centre).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .config import BoxGeometry, GridResolution, Scales


@dataclass
class SceneBatch:
    phi: torch.Tensor                  # (B, 1, X, Y, Z) level set, > 0 in water
    potential: torch.Tensor            # (B, 3, X, Y, Z) vector potential A, u = curl(A)
    pressure: torch.Tensor             # (B, 1, X, Y, Z)
    solid_mask: torch.Tensor           # (1, 1, X, Y, Z) 1 in wall cells (the one-cell border)
    stone_mask: torch.Tensor           # (B, 1, X, Y, Z) 1 inside the stone
    stone_velocity_field: torch.Tensor  # (B, 3, X, Y, Z) velocity of the stone surface, relative to the box (eq. 13)
    roughness: torch.Tensor            # (B,) roughness height k_s, dimensionless
    gravity: torch.Tensor              # (B, 3) gravity direction in the box frame (eq. 3)
    tilt_rate: torch.Tensor            # (B,) Omega_z
    tilt_acceleration: torch.Tensor    # (B,) dOmega_z/dt
    cell_positions: torch.Tensor       # (1, 3, X, Y, Z) cell-centre positions measured from the pivot
    initial_volume: torch.Tensor       # (B,) water volume V0 (eq. 7)
    cell_size: float                   # dimensionless h


NETWORK_INPUT_CHANNELS = 16  # see network_input()
NETWORK_OUTPUT_CHANNELS = 5  # change of A (3), p (1), phi (1)


def smooth_heaviside(phi: torch.Tensor, width: float) -> torch.Tensor:
    """H(phi): 1 in water, 0 above it, smooth over about +-width."""
    return 0.5 * (1.0 + torch.tanh(phi / width))


def cell_centre_positions(grid: GridResolution, box: BoxGeometry, scales: Scales, device) -> torch.Tensor:
    nx, ny, nz = grid.total_cells(box)
    h = grid.cell_size

    def axis(n, size):
        index = torch.arange(n, device=device, dtype=torch.float32)
        return (-size / 2 + (index - 0.5) * h) / scales.length  # index 0 is the wall cell

    x, y, z = torch.meshgrid(axis(nx, box.length), axis(ny, box.height), axis(nz, box.width), indexing="ij")
    return torch.stack([x, y, z]).unsqueeze(0)


def wall_mask(grid: GridResolution, box: BoxGeometry, device) -> torch.Tensor:
    nx, ny, nz = grid.total_cells(box)
    mask = torch.ones(1, 1, nx, ny, nz, device=device)
    mask[:, :, 1:-1, 1:-1, 1:-1] = 0.0
    return mask


def random_scene_batch(batch_size: int, grid: GridResolution, device, scales: Scales = Scales(),
                       generator: Optional[torch.Generator] = None) -> SceneBatch:
    box = scales.box
    h = grid.cell_size / scales.length
    positions = cell_centre_positions(grid, box, scales, device)
    solid = wall_mask(grid, box, device)

    def uniform(low, high, *shape):
        return low + (high - low) * torch.rand(*shape, device=device, generator=generator)

    # tilt about the z axis: gravity seen from inside the box (eq. 3)
    tilt = torch.deg2rad(uniform(-box.max_tilt_degrees, box.max_tilt_degrees, batch_size))
    max_rate = math.radians(box.max_tilt_rate_degrees_per_second) * scales.time
    tilt_rate = uniform(-max_rate, max_rate, batch_size)
    tilt_acceleration = uniform(-max_rate, max_rate, batch_size) * 4.0
    gravity = torch.stack([-torch.sin(tilt), -torch.cos(tilt), torch.zeros_like(tilt)], dim=1)

    # flat, calm water at a random depth, surface at right angles to gravity (eqs. 14-16)
    depth = uniform(box.min_water_depth, box.max_water_depth, batch_size) / scales.length
    up = -gravity.view(batch_size, 3, 1, 1, 1)
    surface_point = torch.zeros(batch_size, 3, 1, 1, 1, device=device)
    surface_point[:, 1, 0, 0, 0] = -box.height / 2 / scales.length + depth
    phi = (up * (surface_point - positions)).sum(dim=1, keepdim=True)
    pressure = torch.clamp(phi, min=0.0)  # hydrostatic: rho g (depth), = phi in dimensionless units

    # small random motion, so the states are not all perfectly still
    potential = 0.01 * torch.randn(batch_size, 3, *phi.shape[2:], device=device, generator=generator)

    # one sphere stone with a random position, size and velocity (eq. 13)
    radius = uniform(box.min_stone_diameter / 2, box.max_stone_diameter / 2, batch_size) / scales.length
    half_size = torch.tensor([box.length, box.height, box.width], device=device) / 2 / scales.length
    centre = (torch.rand(batch_size, 3, device=device, generator=generator) * 2 - 1) * (half_size - radius[:, None])
    distance = (positions - centre.view(batch_size, 3, 1, 1, 1)).norm(dim=1, keepdim=True)
    stone_mask = torch.clamp((radius.view(-1, 1, 1, 1, 1) - distance) / h + 0.5, 0.0, 1.0)
    stone_velocity = uniform(-3.0, 3.0, batch_size, 3) / scales.speed
    stone_spin = uniform(-5.0, 5.0, batch_size, 3) * scales.time
    offset = positions - centre.view(batch_size, 3, 1, 1, 1)
    spin_velocity = torch.cross(stone_spin.view(batch_size, 3, 1, 1, 1).expand_as(offset), offset, dim=1)
    box_velocity = torch.cat([-tilt_rate.view(-1, 1, 1, 1, 1) * positions[:, 1:2],
                              tilt_rate.view(-1, 1, 1, 1, 1) * positions[:, 0:1],
                              torch.zeros_like(positions[:, 2:3]).expand(batch_size, -1, -1, -1, -1)], dim=1)
    stone_velocity_field = stone_velocity.view(batch_size, 3, 1, 1, 1) + spin_velocity - box_velocity

    presets = torch.tensor(box.roughness_presets, device=device) / scales.length
    roughness = presets[torch.randint(len(presets), (batch_size,), device=device, generator=generator)]

    fluid = (1.0 - solid) * (1.0 - stone_mask)
    initial_volume = (smooth_heaviside(phi, 1.5 * h) * fluid).sum(dim=(1, 2, 3, 4)) * h ** 3

    return SceneBatch(phi=phi, potential=potential, pressure=pressure, solid_mask=solid,
                      stone_mask=stone_mask, stone_velocity_field=stone_velocity_field,
                      roughness=roughness, gravity=gravity, tilt_rate=tilt_rate,
                      tilt_acceleration=tilt_acceleration, cell_positions=positions,
                      initial_volume=initial_volume, cell_size=h)


def network_input(scene: SceneBatch) -> torch.Tensor:
    """Stack everything the network sees into (B, 16, X, Y, Z)."""
    batch_size = scene.phi.shape[0]
    spatial = scene.phi.shape[2:]

    def broadcast(values):  # (B, C) -> (B, C, X, Y, Z)
        return values.view(batch_size, -1, 1, 1, 1).expand(-1, -1, *spatial)

    max_roughness = max(BoxGeometry().roughness_presets) / Scales().length
    return torch.cat([
        scene.phi,                                            # 1
        scene.potential,                                      # 3
        scene.pressure,                                       # 1
        scene.solid_mask.expand(batch_size, -1, -1, -1, -1),  # 1
        scene.stone_mask,                                     # 1
        scene.stone_velocity_field,                           # 3
        broadcast(scene.roughness[:, None] / max_roughness),  # 1
        broadcast(scene.gravity),                             # 3
        broadcast(scene.tilt_rate[:, None]),                  # 1
        broadcast(scene.tilt_acceleration[:, None]),          # 1
    ], dim=1)
