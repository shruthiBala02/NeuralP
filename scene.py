"""
scene.py  -  the box, its grid, calm water and the stone as the network sees it

Everything here is dimensionless (see config.Scales): lengths are divided by 0.25 m, speeds by sqrt(g * 0.25 m),
times by 0.25 m / speed. The grid is attached to the box; positions are measured from the tilt pivot (box centre).
"""

import math                                                         # radians
from dataclasses import dataclass                                   # simple containers

import torch                                                        # tensors

from config import BoxGeometry, GridResolution, Scales              # constants and grids
from physics_neural_solver import physical_equations as eq          # starting conditions (eqs. 14-16), smoothed step


@dataclass(frozen=True)
class SceneLimits:
    """The slider ranges of the viewer, in dimensionless units."""
    min_depth: float                                                # water level, lowest
    max_depth: float                                                # water level, highest
    max_tilt: float                                                 # tilt angle limit, radians
    max_tilt_rate: float                                            # tilt speed limit
    min_stone_radius: float                                         # stone radius, smallest
    max_stone_radius: float                                         # stone radius, largest
    min_density_ratio: float                                        # stone density / water density, lowest
    max_density_ratio: float                                        # stone density / water density, highest
    max_throw_speed: float                                          # slingshot speed limit
    half_size: tuple                                                # half the inside size of the box (x, y, z)


def scene_limits(scales: Scales = Scales()) -> SceneLimits:
    box, L = scales.box, scales.length                              # geometry and reference length
    return SceneLimits(
        min_depth=box.min_water_depth / L,                          # 0.10 m -> 0.4
        max_depth=box.max_water_depth / L,                          # 0.35 m -> 1.4
        max_tilt=math.radians(box.max_tilt_degrees),                # 15 degrees in radians
        max_tilt_rate=math.radians(box.max_tilt_rate_degrees_per_second) * scales.time,   # 30 deg/s, dimensionless
        min_stone_radius=box.min_stone_diameter / 2 / L,            # 2.5 cm radius
        max_stone_radius=box.max_stone_diameter / 2 / L,            # 5 cm radius
        min_density_ratio=box.min_stone_density / scales.water.density,   # 300 kg/m^3 relative to water
        max_density_ratio=box.max_stone_density / scales.water.density,   # 3000 kg/m^3 relative to water
        max_throw_speed=box.max_throw_speed / scales.speed,         # 3 m/s, dimensionless
        half_size=(box.length / 2 / L, box.height / 2 / L, box.width / 2 / L),   # (2, 1, 1)
    )


def cell_positions(grid: GridResolution, device, scales: Scales = Scales()) -> torch.Tensor:
    """Centre of every grid cell, measured from the box centre: (1, 3, X, Y, Z)."""
    box, h = scales.box, grid.cell_size                             # geometry and cell size in metres
    nx, ny, nz = grid.total_cells(box)                              # cells including the wall layer

    def axis(n, size):                                              # centres along one direction
        index = torch.arange(n, device=device, dtype=torch.float32)   # 0, 1, ..., n-1 (0 is the wall cell)
        return (-size / 2 + (index - 0.5) * h) / scales.length      # cell centre, dimensionless

    x, y, z = torch.meshgrid(axis(nx, box.length), axis(ny, box.height), axis(nz, box.width), indexing="ij")
    return torch.stack([x, y, z]).unsqueeze(0)                      # (1, 3, X, Y, Z)


def wall_cells(grid: GridResolution, device, scales: Scales = Scales()) -> torch.Tensor:
    """1 in the wall cells (the one-cell border around the box), 0 inside: (1, 1, X, Y, Z)."""
    nx, ny, nz = grid.total_cells(scales.box)                       # cells including the wall layer
    mask = torch.ones(1, 1, nx, ny, nz, device=device)              # start with everything as wall
    mask[:, :, 1:-1, 1:-1, 1:-1] = 0.0                              # the inside is not wall
    return mask


def dimensionless_cell_size(grid: GridResolution, scales: Scales = Scales()) -> float:
    return grid.cell_size / scales.length                           # 2.5 cm -> 0.1,  1.25 cm -> 0.05


def calm_water(positions, depth, tilt, scales: Scales = Scales()):
    """Calm water at depth `depth` (B,) for tilt `tilt` (B,): returns (phi, A, p) (eqs. 14-16)."""
    box_height = scales.box.height / scales.length                  # inside height, dimensionless
    return eq.starting_state(positions, depth, tilt, box_height)    # flat surface, at rest, hydrostatic


def water_volume(phi, solid_mask, stone_mask, h):
    """Water volume in every scene (B,): sum of H(phi) over the cells that can hold water (eq. 7)."""
    can_hold_water = (1.0 - solid_mask) * (1.0 - stone_mask)        # not a wall, not inside the stone
    return (eq.smoothed_step(phi, h) * can_hold_water).sum(dim=(1, 2, 3, 4)) * h ** 3


def stone_fields(positions, stone_centre, stone_velocity, stone_spin, stone_radius, stone_active):
    """The stone as the network sees it:
         stone_mask             (B, 1, X, Y, Z)  1 inside the stone
         stone_velocity_field   (B, 3, X, Y, Z)  velocity of every stone point, relative to the box (eq. 13)
    The stone is integrated in the box frame, so its velocity is already relative to the box:
         velocity at a point x = V_s + omega_s x (x - x_s)
    stone_centre, stone_velocity, stone_spin: (B, 3);  stone_radius, stone_active: (B,)."""
    batch = stone_centre.shape[0]                                   # number of scenes
    offset = positions - stone_centre.view(batch, 3, 1, 1, 1)       # x - x_s for every cell
    distance = offset.norm(dim=1, keepdim=True)                     # distance from the stone centre
    inside = (distance < stone_radius.view(batch, 1, 1, 1, 1)).float()   # 1 inside the sphere
    stone_mask = inside * stone_active.view(batch, 1, 1, 1, 1).float()   # no stone if it is not active
    spin = stone_spin.view(batch, 3, 1, 1, 1).expand_as(offset)     # spin at every cell
    velocity = stone_velocity.view(batch, 3, 1, 1, 1) + torch.cross(spin, offset, dim=1)   # V_s + omega x (x - x_s)
    return stone_mask, velocity * stone_mask                        # velocity only inside the stone
