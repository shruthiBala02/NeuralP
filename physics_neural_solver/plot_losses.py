"""
plot_losses.py  -  draw how every loss term changes during training

    python -m physics_neural_solver.plot_losses --run coarse_small      # reads outputs/runs/coarse_small/log.csv

Writes outputs/runs/<run>/losses.png: nine small panels - the total loss and each of the 8 equations - each with
its own logarithmic axis (the terms differ by orders of magnitude, so one shared axis would hide most of them).
In every panel: the faint line is every logged value, the solid line is a smoothed version, the thin vertical
lines mark where the curriculum stages (calm -> tilt -> stone) begin, and the number at the end is the latest value.
train.py also calls this at every checkpoint, so the picture stays up to date while training runs.
"""

import argparse                                                     # command-line options
import csv                                                          # reading log.csv
from pathlib import Path                                            # file paths

import matplotlib                                                   # plotting
matplotlib.use("Agg")                                               # draw to files (no screen needed on the cluster)
import matplotlib.pyplot as plt                                     # noqa: E402  (after choosing the backend)
import numpy as np                                                  # noqa: E402  numbers

SURFACE = "#fcfcfb"                                                 # background
INK = "#0b0b0b"                                                     # titles and numbers
SECONDARY_INK = "#52514e"                                           # axis labels and tick labels
GRID = "#e4e3df"                                                    # hairline grid, one step off the background
LINE = "#2a78d6"                                                    # the data (one series per panel)

PANELS = [                                                          # (column in log.csv, panel title)
    ("total", "Total loss (weighted sum)"),
    ("momentum (eq. 2)", "Momentum, Navier-Stokes (eq. 2)"),
    ("level set (eq. 5)", "Surface transport (eq. 5)"),
    ("eikonal (eq. 6)", "Eikonal, sharp surface (eq. 6)"),
    ("volume (eq. 7)", "Volume conservation (eq. 7)"),
    ("no shear (eq. 9)", "Free-surface no-shear (eq. 9)"),
    ("no penetration (eq. 10)", "Wall no-penetration (eq. 10)"),
    ("wall law (eq. 11)", "Wall friction, wall law (eq. 11)"),
    ("stone boundary (eq. 13)", "Water-stone boundary (eq. 13)"),
]


def read_log(path: Path):
    """Columns of log.csv as numbers (and the stage names as text)."""
    with path.open() as f:
        rows = list(csv.DictReader(f))                              # one dict per log line
    if not rows:
        raise ValueError(f"{path} has no data yet")
    columns = {name: np.array([float(r[name]) for r in rows]) for name, _ in PANELS}   # the loss columns
    columns["step"] = np.array([int(r["step"]) for r in rows])      # training step of each line
    columns["stage"] = [r["stage"] for r in rows]                   # calm / tilt / stone
    return columns


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average over `window` log lines, computed on the logarithm (a fair average for values on a log axis)."""
    if len(values) < 3 or window < 2:
        return values                                               # too short to smooth
    logs = np.log10(np.clip(values, 1e-30, None))                   # work in log space
    kernel = np.ones(window) / window                               # equal weights
    padded = np.concatenate([np.full(window - 1, logs[0]), logs])   # repeat the first value so the start is not cut off
    return 10 ** np.convolve(padded, kernel, mode="valid")          # back from log space


def stage_starts(steps, stages):
    """(step, stage name) where each new stage begins."""
    starts, previous = [], None
    for step, stage in zip(steps, stages):
        if stage != previous:                                       # a new stage
            starts.append((step, stage))
            previous = stage
    return starts


def plot(run_dir: Path) -> Path:
    data = read_log(run_dir / "log.csv")                            # the numbers
    steps = data["step"]
    window = max(1, len(steps) // 40)                               # smooth over ~2.5 % of the run
    starts = stage_starts(steps, data["stage"])                     # where calm / tilt / stone begin

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": SECONDARY_INK, "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK})
    figure, axes = plt.subplots(3, 3, figsize=(13, 9.5), facecolor=SURFACE, sharex=True)
    for ax, (column, title) in zip(axes.flat, PANELS):              # one panel per term
        values = data[column]
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
        positive = values > 0                                       # a log axis cannot show zeros
        if not positive.any():                                      # e.g. no stones yet: the term is exactly 0
            ax.text(0.5, 0.5, "0 so far", transform=ax.transAxes, ha="center", va="center", color=SECONDARY_INK)
            ax.set_yticks([])
        else:
            x, y = steps[positive], values[positive]                # only the values a log axis can show
            ax.plot(x, y, color=LINE, alpha=0.25, linewidth=0.8)    # every logged value
            ax.plot(x, smooth(y, window), color=LINE, linewidth=1.6)   # smoothed trend
            ax.set_yscale("log")
            last = values[positive][-1]                             # latest value, written at the end of the line
            ax.annotate(f"{last:.2e}", xy=(steps[positive][-1], last), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK, annotation_clip=False)
        ax.grid(True, which="major", color=GRID, linewidth=0.6, linestyle="-")   # solid hairline grid
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)                     # no box around the panel
        for step, stage in starts[1:]:                              # stage boundaries (not at step 0)
            ax.axvline(step, color="#bdbcb6", linewidth=0.8, zorder=0)
        for step, stage in starts:                                  # stage names along the top of the panel
            ax.text(step, 1.0, f" {stage}", transform=ax.get_xaxis_transform(), va="bottom",
                    fontsize=7.5, color=SECONDARY_INK)
    for ax in axes[-1]:
        ax.set_xlabel("training step")                              # only the bottom row
    for ax in axes[:, 0]:
        ax.set_ylabel("loss (log scale)")                           # only the left column

    figure.suptitle(f"Training losses - run '{run_dir.name}', {steps[-1]:,} steps", x=0.01, y=0.99, ha="left",
                    color=INK, fontsize=12)
    figure.text(0.01, 0.945, "faint line: every logged value   solid line: smoothed   vertical lines: start of a "
                "curriculum stage   number: latest value", ha="left", fontsize=8.5, color=SECONDARY_INK)
    figure.tight_layout(rect=(0, 0, 0.985, 0.93))
    out_path = run_dir / "losses.png"
    figure.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot the training losses of a run.")
    parser.add_argument("--run", default="coarse_small", help="run name under outputs/runs/")
    args = parser.parse_args()
    print(f"Wrote {plot(Path('outputs/runs') / args.run)}")


if __name__ == "__main__":
    main()
