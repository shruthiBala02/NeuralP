"""
All physical constants, box geometry and grid settings in one place.

Everything follows the equation section of PLAN.md (simple version: box only,
water only, tilt about one axis). Inside the code all quantities are made
dimensionless with the scales in `Scales` (length = reference depth,
speed = sqrt(g * length), time = length / speed, pressure = rho * g * length).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WaterConstants:
    density: float = 998.2                # kg/m^3, water at 20 C
    kinematic_viscosity: float = 1.004e-6  # m^2/s, water at 20 C
    gravity: float = 9.81                  # m/s^2
    smagorinsky_constant: float = 0.1      # C_s (eq. 4), to be tuned once in the classical solver
    von_karman_constant: float = 0.41      # kappa (eq. 11)
    log_law_offset: float = 5.2            # B (eq. 11)
    roughness_fit_constant: float = 0.3    # the 0.3 in eq. 12


@dataclass(frozen=True)
class BoxGeometry:
    length: float = 1.0   # x, metres (inside size)
    height: float = 0.5   # y (up)
    width: float = 0.5    # z
    default_water_depth: float = 0.25
    min_water_depth: float = 0.10
    max_water_depth: float = 0.35
    max_tilt_degrees: float = 15.0
    max_tilt_rate_degrees_per_second: float = 30.0
    min_stone_diameter: float = 0.05
    max_stone_diameter: float = 0.10
    min_stone_density: float = 300.0
    max_stone_density: float = 3000.0
    roughness_presets: tuple = (0.0, 0.001, 0.005)  # k_s in metres: smooth / medium / rough


@dataclass(frozen=True)
class GridResolution:
    """A uniform grid over the box interior, plus one layer of wall cells on every side."""
    name: str
    cell_size: float  # metres

    def interior_cells(self, box: BoxGeometry) -> tuple:
        return (round(box.length / self.cell_size),
                round(box.height / self.cell_size),
                round(box.width / self.cell_size))

    def total_cells(self, box: BoxGeometry) -> tuple:
        nx, ny, nz = self.interior_cells(box)
        return (nx + 2, ny + 2, nz + 2)  # + one wall cell on each side


COARSE = GridResolution("coarse", 0.025)   # 40 x 20 x 20 interior cells
FINE = GridResolution("fine", 0.0125)      # 80 x 40 x 40 interior cells
GRIDS = {"coarse": COARSE, "fine": FINE}


@dataclass(frozen=True)
class Scales:
    """Reference scales used to make the equations dimensionless."""
    water: WaterConstants = WaterConstants()
    box: BoxGeometry = BoxGeometry()

    @property
    def length(self) -> float:
        return self.box.default_water_depth  # 0.25 m

    @property
    def speed(self) -> float:
        return math.sqrt(self.water.gravity * self.length)  # ~1.57 m/s

    @property
    def time(self) -> float:
        return self.length / self.speed  # ~0.16 s

    @property
    def reynolds_number(self) -> float:
        return self.speed * self.length / self.water.kinematic_viscosity  # ~3.9e5

    @property
    def dimensionless_viscosity(self) -> float:
        return 1.0 / self.reynolds_number
