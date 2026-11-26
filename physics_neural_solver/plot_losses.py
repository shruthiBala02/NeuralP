"""
plot_losses.py  -  draw how every loss term changes during training

    python -m physics_neural_solver.plot_losses --run coarse_small      # reads outputs/runs/coarse_small/log.csv

Writes two pictures into outputs/runs/<run>/:
    losses.png               ONE graph: the total loss (thick black) and the 9 terms (one colour each), smoothed,
                             on one logarithmic axis; thin vertical lines mark the curriculum stages (calm -> tilt -> stone)
    losses_by_equation.png   the same ten curves in separate panels (each with its own axis), with every logged value
A term that is exactly 0 (e.g. the water-stone loss before any stone is thrown) cannot be drawn on a log axis,
so its line starts when it first becomes non-zero.
train.py also calls this at every checkpoint, so the pictures stay up to date while training runs.
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
EQUATION_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",      # one fixed colour per equation (eqs. 2, 5, 6, 7,
                    "#e87ba4", "#008300", "#6250d6", "#e34948",      #  9, 10, 11, 13 and the energy rule),
                    "#5c5a55"]                                      #  never re-ordered

PANELS = [                                                          # (column in log.csv, panel title)
    ("total", "Total loss (weighted sum)"),
    ("momentum (eq. 2)", "Momentum, Navier-Stokes (eq. 2)"),
    ("level set (eq. 5)", "Surface transport (eq. 5, exact)"),
    ("eikonal (eq. 6)", "Eikonal, sharp surface (eq. 6, exact)"),
    ("volume (eq. 7)", "Volume conservation (eq. 7, exact)"),
    ("no shear (eq. 9)", "Free-surface no-shear (eq. 9)"),
    ("no penetration (eq. 10)", "Wall no-penetration (eq. 10, exact)"),
    ("wall law (eq. 11)", "Wall friction, wall law (eq. 11)"),
    ("stone boundary (eq. 13)", "Water-stone boundary (eq. 13)"),
    ("energy (rule)", "Energy can't grow by itself (rule)"),
]


def read_log(path: Path):
    """Columns of log.csv as numbers (and the stage names as text)."""
    with path.open() as f:
        rows = list(csv.DictReader(f))                              # one dict per log line
    if not rows:
        raise ValueError(f"{path} has no data yet")
    columns = {name: np.array([float(r[name]) if name in r else 0.0 for r in rows]) for name, _ in PANELS}   # the loss columns
                                                                    # (older runs have no energy column: shown as 0)
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


def plot_panels(run_dir: Path) -> Path:
    data = read_log(run_dir / "log.csv")                            # the numbers
    steps = data["step"]
    window = max(1, len(steps) // 40)                               # smooth over ~2.5 % of the run
    starts = stage_starts(steps, data["stage"])                     # where calm / tilt / stone begin

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": SECONDARY_INK, "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK})
    figure, axes = plt.subplots(2, 5, figsize=(19, 7.5), facecolor=SURFACE, sharex=True)
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
    out_path = run_dir / "losses_by_equation.png"
    figure.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    return out_path


def plot_combined(run_dir: Path) -> Path:
    """All ten curves in one graph, on one logarithmic axis."""
    data = read_log(run_dir / "log.csv")                            # the numbers
    steps = data["step"]
    window = max(1, len(steps) // 40)                               # smooth over ~2.5 % of the run
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": SECONDARY_INK, "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK})
    figure, ax = plt.subplots(figsize=(12, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    curves = []                                                     # (latest value, label, line) for the legend order
    for (column, title), colour in zip(PANELS, ["#0b0b0b"] + EQUATION_COLOURS):   # total in black, then the equations
        values = data[column]
        positive = values > 0                                       # a log axis cannot show zeros
        if not positive.any():
            curves.append((0.0, f"{title}  (0 so far)", None))      # still listed in the legend
            continue
        x, y = steps[positive], smooth(values[positive], window)    # smoothed curve
        width = 2.4 if column == "total" else 1.6                   # the total stands out
        (line,) = ax.plot(x, y, color=colour, linewidth=width, zorder=3 if column == "total" else 2)
        curves.append((float(y[-1]), f"{title}   {values[positive][-1]:.1e}", line))

    for step, stage in stage_starts(steps, data["stage"])[1:]:      # stage boundaries
        ax.axvline(step, color="#bdbcb6", linewidth=0.8, zorder=0)
    for step, stage in stage_starts(steps, data["stage"]):          # stage names along the top
        ax.text(step, 1.0, f" {stage}", transform=ax.get_xaxis_transform(), va="bottom", fontsize=9, color=SECONDARY_INK)

    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6, linestyle="-")   # solid hairline grid
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (log scale)")

    curves.sort(key=lambda item: -item[0])                          # legend from the highest to the lowest latest value
    handles = [line if line is not None else plt.Line2D([], [], color="none") for _, _, line in curves]
    legend = ax.legend(handles, [label for _, label, _ in curves], loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       frameon=False, fontsize=9, labelcolor=INK, handlelength=2.2,
                       title="latest value (smoothed curves)", title_fontsize=9)
    legend.get_title().set_color(SECONDARY_INK)
    legend._legend_box.align = "left"

    figure.suptitle(f"Training losses - run '{run_dir.name}', {steps[-1]:,} steps", x=0.01, y=0.98, ha="left",
                    color=INK, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = run_dir / "losses.png"
    figure.savefig(out_path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot(run_dir: Path) -> Path:
    """Write both pictures; returns the path of the single-graph one."""
    plot_panels(run_dir)                                            # ten panels
    return plot_combined(run_dir)                                   # one graph


def main():
    parser = argparse.ArgumentParser(description="Plot the training losses of a run.")
    parser.add_argument("--run", default="coarse_small", help="run name under outputs/runs/")
    args = parser.parse_args()
    run_dir = Path("outputs/runs") / args.run
    plot(run_dir)
    print(f"Wrote {run_dir / 'losses.png'} and {run_dir / 'losses_by_equation.png'}")


if __name__ == "__main__":
    main()
