"""
Tests of export_for_browser.py: the files the viewer loads must be complete and consistent with the network.
"""

import json                                                         # model.json
import os                                                           # changing folder
import sys                                                          # command-line arguments

import numpy as np                                                  # reading the binary files

from physics_neural_solver import export_for_browser                # the exporter
from physics_neural_solver.model import PhysicsNeuralSolver, count_weights


def test_test_network_export_is_complete_and_consistent(tmp_path, monkeypatch):
    here = os.getcwd()
    os.chdir(tmp_path)                                              # write into a temporary folder
    try:
        monkeypatch.setattr(sys, "argv", ["export", "--test-network", "--grid", "coarse"])
        export_for_browser.main()
        folder = tmp_path / "outputs" / "export" / "test_network_coarse"
        manifest = json.loads((folder / "model.json").read_text())
        weights = np.fromfile(folder / "weights.bin", dtype=np.float32)
        assert weights.size == count_weights(PhysicsNeuralSolver("small"))       # every weight is there
        last = manifest["layers"][-1]
        assert last["offset"] + last["count"] == weights.size                    # the layout covers the whole file
        assert manifest["grid"]["cells_with_walls"] == [42, 22, 22]
        assert manifest["input_order"][0] == "phi" and len(manifest["input_order"]) == 15
        test = np.fromfile(folder / "test_case.bin", dtype=np.float32)
        cells = 42 * 22 * 22
        padded = 48 * 24 * 24                                       # padded to a multiple of 2^3
        assert test.size == 14 * cells + 19 * padded                # state, expected state and velocity, input (15),
                                                                    # output (4), phi after re-distancing
        assert manifest["output_order"] == ["dA_x", "dA_y", "dA_z", "dp"]   # phi is carried, not predicted
        assert manifest["test_case"]["initial_volume"] > 0
        assert np.isfinite(test).all()
    finally:
        os.chdir(here)
