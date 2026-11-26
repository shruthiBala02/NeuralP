"""
config.py  -  all physical constants, box geometry and grid settings in one place (values from PLAN.md)

Inside the solver everything is dimensionless, using the reference scales in `Scales`:
    length = 0.25 m (default water depth),  speed = sqrt(g * length),  time = length / speed,  pressure = rho g length
"""

import math                                                         # sqrt
from dataclasses import dataclass                                   # simple read-only containers


@dataclass(frozen=True)
class WaterConstants:                                               # water at 20 C
    density: float = 998.2                                          # rho, kg/m^3
    kinematic_viscosity: float = 1.004e-6                           # nu, m^2/s
    gravity: float = 9.81                                           # |g|, m/s^2
    smagorinsky_constant: float = 0.1                               # C_s (eq. 4)
    von_karman_constant: float = 0.41                               # kappa (eq. 11, wall law)
    log_law_offset: float = 5.2                                     # B (eq. 11, wall law)


@dataclass(frozen=True)
class BoxGeometry:                                                  # the box and the slider ranges
    length: float = 1.0                                             # inside length along x, metres
    height: float = 0.5                                             # inside height along y (up), metres
    width: float = 0.5                                              # inside width along z, metres
    default_water_depth: float = 0.25                               # h_0 default, metres
    min_water_depth: float = 0.10                                   # water-level slider, lowest
    max_water_depth: float = 0.35                                   # water-level slider, highest
    max_tilt_degrees: float = 15.0                                  # tilt slider, +-
    max_tilt_rate_degrees_per_second: float = 30.0                  # how fast the box can tilt
    min_stone_diameter: float = 0.05                                # stone size slider, metres
    max_stone_diameter: float = 0.10
    min_stone_density: float = 300.0                                # stone density slider, kg/m^3
    max_stone_density: float = 3000.0
    max_throw_speed: float = 3.0                                    # slingshot throw, m/s


@dataclass(frozen=True)
class GridResolution:
    """A uniform grid over the inside of the box, plus one layer of wall cells on every side."""
    name: str                                                       # "coarse" or "fine"
    cell_size: float                                                # metres

    def interior_cells(self, box: BoxGeometry) -> tuple:
        return (round(box.length / self.cell_size),                 # cells along x
                round(box.height / self.cell_size),                 # cells along y
                round(box.width / self.cell_size))                  # cells along z

    def total_cells(self, box: BoxGeometry) -> tuple:
        nx, ny, nz = self.interior_cells(box)                       # inside cells
        return (nx + 2, ny + 2, nz + 2)                             # + one wall cell on each side


COARSE = GridResolution("coarse", 0.025)                            # 40 x 20 x 20 inside cells (2.5 cm)
FINE = GridResolution("fine", 0.0125)                               # 80 x 40 x 40 inside cells (1.25 cm)
GRIDS = {"coarse": COARSE, "fine": FINE}                            # choose by name


@dataclass(frozen=True)
class Scales:
    """Reference scales that make the equations dimensionless."""
    water: WaterConstants = WaterConstants()                        # physical constants
    box: BoxGeometry = BoxGeometry()                                # geometry

    @property
    def length(self) -> float:
        return self.box.default_water_depth                         # 0.25 m

    @property
    def speed(self) -> float:
        return math.sqrt(self.water.gravity * self.length)          # ~1.57 m/s

    @property
    def time(self) -> float:
        return self.length / self.speed                             # ~0.16 s

    @property
    def reynolds_number(self) -> float:
        return self.speed * self.length / self.water.kinematic_viscosity   # ~3.9e5

    @property
    def dimensionless_viscosity(self) -> float:
        return 1.0 / self.reynolds_number                           # nu in dimensionless units
