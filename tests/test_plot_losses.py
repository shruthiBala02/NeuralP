"""Tests of plot_losses.py: it must draw losses.png from a log, including terms that are still exactly zero."""

import csv                                                          # writing a small log

from physics_neural_solver.plot_losses import PANELS, plot, smooth  # the plotting code


def test_plot_is_written_from_a_log(tmp_path):
    names = [column for column, _ in PANELS]                        # total + the 8 equations
    with (tmp_path / "log.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "stage"] + names)
        for i in range(1, 121):                                     # 120 log lines through the three stages
            stage = "calm" if i < 40 else "tilt" if i < 80 else "stone"
            values = [1e-2 / i ** 0.5] * len(names)
            values[-1] = 0.0 if stage != "stone" else 1e-3          # the stone term is 0 before stones exist
            writer.writerow([i * 50, stage] + values)
    image = plot(tmp_path)
    assert image.exists() and image.stat().st_size > 20_000         # the single graph (losses.png)
    panels = tmp_path / "losses_by_equation.png"
    assert panels.exists() and panels.stat().st_size > 20_000       # the nine-panel version


def test_smoothing_keeps_a_constant_constant():
    import numpy as np
    values = np.full(50, 3e-4)
    assert np.allclose(smooth(values, 5), values)                   # averaging a constant changes nothing
