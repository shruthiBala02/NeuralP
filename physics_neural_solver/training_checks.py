"""
training_checks.py  -  short tests of the network DURING training (an early warning)

Every `check_every` steps (train.py) the current network runs a few seconds of water on its own, exactly as the
viewer would, and a few numbers are written to outputs/runs/<run>/checks.csv and printed:
    still water   fastest water (m/s) and the largest volume change      (should stay ~0)
    sloshing      tilt to 10 degrees and hold: the sloshing period (s)   (theory 1.40 s) and how far the water swings
Without this, a problem only shows up when a kept network is evaluated (every 200k steps, hours later).
The analytical answers are used only for these printed checks, never in the training loss.
"""

import csv                                                          # checks.csv
from pathlib import Path                                            # file paths

from physics_neural_solver.evaluate import test_slosh, test_still   # the same tests as evaluate.py, shorter

COLUMNS = ["step", "still_fastest_m_s", "still_volume_change", "slosh_period_s", "slosh_theory_s", "slosh_swing_m"]


def run_checks(model, settings: dict, device, seconds: float) -> dict:
    """Run the two short tests with the current network; returns the numbers."""
    was_training = model.training                                   # (the network has no dropout/batch norm, but be tidy)
    model.eval()
    still = test_still(model, settings, device, seconds=seconds)    # water at rest, level box
    slosh = test_slosh(model, settings, device, seconds=seconds)    # tilt to 10 degrees and hold
    model.train(was_training)
    heights = [y for t, y in zip(slosh["series_time_s"], slosh["series_wall_height_m"]) if t > 1.5 and y == y]
    swing = (max(heights) - min(heights)) if heights else float("nan")   # how far the water at the wall moves up and down
    return {"still_fastest_m_s": still["max_water_speed_m_s"], "still_volume_change": still["max_volume_error"],
            "slosh_period_s": slosh["period_s"], "slosh_theory_s": slosh["theory_period_s"], "slosh_swing_m": swing}


def write_checks(run_dir: Path, step: int, results: dict):
    """Append one line to checks.csv (with the column titles the first time)."""
    path = run_dir / "checks.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(COLUMNS)
        writer.writerow([step] + [f"{results[name]:.4g}" for name in COLUMNS[1:]])


def describe(results: dict) -> str:
    """One readable line for the terminal."""
    return (f"still water: fastest {results['still_fastest_m_s']:.3f} m/s, volume change "
            f"{100 * results['still_volume_change']:.2f} %  |  slosh: period {results['slosh_period_s']:.2f} s "
            f"(theory {results['slosh_theory_s']:.2f} s), swing {100 * results['slosh_swing_m']:.1f} cm")
