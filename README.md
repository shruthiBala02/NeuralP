# neuralP

Cluster code for a **physics-informed neural solver for 3D water** in a tilting box
(water only, one stone). The network is trained only with the physics equations
(3D incompressible Navier–Stokes + a level-set equation for the free surface); no simulation data.
The full plan and all equations are in `PLAN.md` (kept outside this repository).

This repository contains **only the code that runs on the cluster, plus its outputs**.
The browser viewer (Three.js) lives elsewhere.

## Current stage: 0, timing benchmark

Before any real training we measure, on the cluster GPU, how long one training step takes
(network → physics loss with all 8 equation terms → backward → Adam update) and how long
one solver step takes (what the browser will run every frame), for:

| setting | values |
|---|---|
| grid | coarse 2.5 cm (40×20×20 interior cells), fine 1.25 cm (80×40×40) |
| network | 3D U-Net, small / medium / large (16 / 32 / 64 base channels) |
| batch size | coarse 8–64, fine 2–16 |
| precision | fp32 (strict), tf32 (float32 on tensor cores), bf16 (network in bfloat16, physics in float32) |

It does not train the real model; the states are random calm-water scenes.

## Commands

Run from this folder on the cluster.

```bash
# 0. get the code (first time)
git clone https://github.com/shruthiBala02/NeuralP.git
cd NeuralP

# 1. one-time setup (Python venv + PyTorch for CUDA 12.8)
bash scripts/setup_environment.sh

# 2a. on an interactive GPU node:
bash scripts/run_benchmark.sh --quick     # 1-minute code check first
bash scripts/run_benchmark.sh             # full benchmark (estimate: 30–60 min)

# 2b. or as a SLURM job (edit the partition line first if needed):
sbatch scripts/benchmark.sbatch

# 3. send the results back
git add outputs/
git commit -m "Benchmark results"
git push
```

Results go to `outputs/benchmark/<date-time>/`:

| file | content |
|---|---|
| `summary.md` | the easiest to read: all timings + projected training hours |
| `results.csv`, `results.json` | the same numbers, machine-readable |
| `environment.json` | GPU, driver, PyTorch and CUDA versions |
| `self_tests.json` | correctness checks (divergence of u = curl A is ~0, loss is finite) |

## Code layout

```
neuralp/
├── config.py             physical constants, box size, coarse / fine grids, dimensionless scales
├── grid.py               finite differences on the staggered (MAC) grid; u = curl(A)
├── scene.py              random water states (calm water, tilt, stone) and the network input
├── network.py            the 3D U-Net (predicts the change of A, p and phi)
├── physics_loss.py       the loss: residuals of equations 2, 5, 6, 7, 9, 10, 11, 13 of the plan
├── check_environment.py  prints GPU / PyTorch / CUDA information
└── benchmark.py          stage 0 timing benchmark
scripts/
├── setup_environment.sh
├── run_benchmark.sh
└── benchmark.sbatch
outputs/                  results (committed); checkpoints are never committed
```
