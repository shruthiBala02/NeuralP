"""
Tests of evaluate.py: the evaluation must not be fooled by water that simply never moves.
(An untrained network outputs zero change, so its water stays frozen. Frozen water passes "still", "sink" and
"float" - those mostly test the exact stone code - so "slosh" and "splash" must be the ones that catch it.)
"""

import os                                                           # changing folder
import sys                                                          # command-line arguments
from dataclasses import asdict                                      # settings -> dict

import json                                                         # results.json
import torch                                                        # saving the checkpoint

from physics_neural_solver import evaluate                          # the evaluation
from physics_neural_solver.model import PhysicsNeuralSolver         # the network
from physics_neural_solver.train import TrainingSettings            # its settings


def test_frozen_water_fails_the_water_tests(tmp_path, monkeypatch):
    here = os.getcwd()
    os.chdir(tmp_path)                                              # write into a temporary folder
    try:
        run = tmp_path / "outputs" / "runs" / "frozen"
        run.mkdir(parents=True)
        torch.save({"model": PhysicsNeuralSolver("small").state_dict(), "settings": asdict(TrainingSettings()),
                    "step": 0}, run / "checkpoint.pt")              # an untrained network: the water never moves
        monkeypatch.setattr(sys, "argv", ["evaluate", "--run", "frozen", "--quick", "--device", "cpu"])
        evaluate.main()
        out = tmp_path / "outputs" / "evaluation" / "frozen"
        results = json.loads((out / "results.json").read_text())
        assert not results["slosh (C2)"]["pass"]                    # no sloshing: caught
        assert not results["splash (stone)"]["pass"]                # the stone doesn't move the water: caught
        assert (out / "evaluation.png").exists() and (out / "summary.md").exists()
        assert "series" not in (out / "summary.md").read_text()     # the time series stay out of the summary
    finally:
        os.chdir(here)
