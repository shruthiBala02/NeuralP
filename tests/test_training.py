"""
Tests of the training machinery: the pool, the curriculum, the loss balancing, checkpoints - and the key question
whether the physics loss can actually be LEARNED (a small network must reduce it on a fixed batch).
"""

import math                                                         # pi
import os                                                           # changing folder for the checkpoint test

import torch                                                        # tensors

from config import COARSE, Scales                                   # grid and reference scales
from physics_neural_solver import physical_equations as eq          # the loss
from physics_neural_solver.model import PhysicsNeuralSolver, WaterState   # the network
from physics_neural_solver.train import TrainingSettings, learning_rate_at, stage_at, train, weighted_total
from physics_neural_solver.training_pool import TrainingPool        # training states

DT = (1 / 60) / Scales().time                                       # one frame, dimensionless


def test_pool_starts_calm_and_samples_batches():
    pool = TrainingPool(8, COARSE, "cpu", DT)
    indices, state, box = pool.sample(4)
    assert len(set(indices.tolist())) == 4                          # four different scenes
    assert state.phi.shape[0] == 4 and box.stone_mask.shape[0] == 4
    assert torch.all(state.potential == 0)                          # calm water at the start
    assert torch.all(pool.simulation.tilt == 0)                     # stage "calm": level box


def test_curriculum_actions_happen_in_the_right_stages():
    pool = TrainingPool(64, COARSE, "cpu", DT, seed=1)
    model = PhysicsNeuralSolver("small")
    for stage, expect_tilt, expect_stone in (("calm", False, False), ("tilt", True, False), ("stone", True, True)):
        pool.set_stage(stage)
        for _ in range(40):                                         # many steps so that the random actions happen
            indices, state, box = pool.sample(16)
            with torch.no_grad():
                pool.put_back(indices, model(state, box, pool.simulation.h))
        tilted = bool((pool.simulation.tilt_target != 0).any())
        stones = bool(pool.simulation.stones.active.any())
        assert tilted == expect_tilt or (stage != "calm"), stage    # calm: never tilted
        if stage == "calm":
            assert not tilted and not stones
        if stage == "tilt":
            assert tilted and not stones
        if stage == "stone":
            assert stones


def test_broken_scenes_are_reset():
    pool = TrainingPool(4, COARSE, "cpu", DT)
    pool.simulation.water.phi[2, 0, 5, 5, 5] = float("nan")         # break one scene
    assert bool(pool.simulation.broken_scenes()[2])
    indices = torch.tensor([2])
    _, state, _ = pool.sample(4)
    broken = WaterState(pool.simulation.water.potential[indices], pool.simulation.water.pressure[indices],
                        pool.simulation.water.phi[indices])
    pool.put_back(indices, broken)
    assert torch.isfinite(pool.simulation.water.phi[2]).all()       # it was reset to calm water


def test_weighted_total_uses_the_importance():
    total = weighted_total({"a": torch.tensor(2.0), "b": torch.tensor(3.0)}, {"a": 1.0, "b": 20.0})
    assert float(total) == 62.0                                     # 1 x 2 + 20 x 3


def test_schedule_and_stages():
    settings = TrainingSettings()
    assert stage_at(0, settings.stage_steps) == "calm"
    assert stage_at(settings.stage_steps["calm"], settings.stage_steps) == "tilt"
    assert stage_at(settings.total_steps - 1, settings.stage_steps) == "stone"
    assert abs(learning_rate_at(0, settings) - settings.learning_rate) < 1e-12
    assert abs(learning_rate_at(settings.total_steps, settings) - settings.final_learning_rate) < 1e-12


def test_network_learns_the_hydrostatic_pressure_from_the_physics_alone():
    # calm water with the pressure MISSING: the only way to satisfy Navier-Stokes is to output the hydrostatic
    # pressure. Nobody tells the network that answer - it must find it from the equations.
    torch.manual_seed(0)
    pool = TrainingPool(4, COARSE, "cpu", DT)
    pool.set_stage("tilt")
    pool.reset(torch.arange(4))                                     # calm water in tilted boxes
    _, state, box = pool.sample(2)
    no_pressure = WaterState(state.potential, torch.zeros_like(state.pressure), state.phi)
    h, positions, volume = pool.simulation.h, pool.simulation.positions, pool.simulation.initial_volume[:2]
    model = PhysicsNeuralSolver("small")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    importance = TrainingSettings().importance

    def terms():
        return eq.loss_terms(no_pressure, model(no_pressure, box, h), box, positions, volume, h, DT)

    first = float(terms()["momentum (eq. 2)"].detach())
    for _ in range(150):                                            # train on this one batch, like training does
        loss = weighted_total(terms(), importance)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        prediction = model(no_pressure, box, h)
        last = float(terms()["momentum (eq. 2)"])
    assert last < 0.02 * first, f"momentum loss {first:.3e} -> {last:.3e}: not learning"   # error down by 50x
    water = (state.phi > 0.2) & (box.solid_mask == 0)               # more than 5 cm under the surface (away from the surface cells)
    learned, hydrostatic = prediction.pressure[water], state.pressure[water]
    assert (learned - hydrostatic).abs().mean() < 0.1 * hydrostatic.abs().mean()   # it found p = rho g depth (within 10 %)


def test_training_runs_and_resumes(tmp_path):
    here = os.getcwd()
    os.chdir(tmp_path)                                              # write outputs into a temporary folder
    try:
        settings = TrainingSettings(run_name="t", pool_size=4, batch_size=2, device="cpu",
                                    stage_steps={"calm": 3, "tilt": 3, "stone": 3})
        settings.log_every, settings.checkpoint_every = 3, 3
        train(settings)                                             # 9 steps
        checkpoint = torch.load("outputs/runs/t/checkpoint.pt", weights_only=False)
        assert checkpoint["step"] == 9
        settings.stage_steps = {"calm": 3, "tilt": 3, "stone": 6}   # 3 more steps
        train(settings, resume=True)
        assert torch.load("outputs/runs/t/checkpoint.pt", weights_only=False)["step"] == 12
        lines = open("outputs/runs/t/log.csv").read().strip().splitlines()
        assert len(lines) == 1 + 4                                  # header + 4 log lines (steps 3, 6, 9, 12)
    finally:
        os.chdir(here)
