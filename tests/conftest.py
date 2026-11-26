"""Shared set-up for the tests: makes the repository folder importable and provides the grid helpers."""

import sys                                                          # the import path
from pathlib import Path                                            # file paths

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # so that `import config` works from tests/

import pytest                                                       # test framework
import torch                                                        # tensors

from config import COARSE, FINE, Scales                             # grids and reference scales
from scene import cell_positions, dimensionless_cell_size, wall_cells   # grid helpers

torch.manual_seed(0)                                                # the same random numbers every run


@pytest.fixture(params=[COARSE, FINE], ids=["coarse", "fine"])
def grid(request):
    return request.param                                            # every grid test runs on both grids


@pytest.fixture
def geometry(grid):
    """(positions, wall mask, cell size h) for the grid, all dimensionless."""
    return cell_positions(grid, "cpu"), wall_cells(grid, "cpu"), dimensionless_cell_size(grid)


@pytest.fixture
def scales():
    return Scales()                                                 # reference scales
