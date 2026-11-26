"""
benchmark.py  -  how fast is training and running the solver on THIS machine? (does not train the real model)

    python -m physics_neural_solver.benchmark                  # coarse + fine, small/medium/large, fp32/tf32/bf16
    python -m physics_neural_solver.benchmark --quick          # a tiny version, to check the code

For every combination it runs real training steps (pool -> network -> all 8 loss terms -> backward -> Adam ->
back into the pool) and one solver step at batch 1 without gradients (what the viewer runs every frame).
Output: outputs/benchmark/<date-time>/summary.md and results.json
"""

import argparse                                                     # command-line options
import json                                                         # results file
import statistics                                                   # median
import time                                                         # timing
from datetime import datetime                                       # folder name
from pathlib import Path                                            # file paths

import torch                                                        # tensors

from check_environment import environment_report                    # GPU / PyTorch information
from config import GRIDS, Scales                                    # grids and reference scales
from physics_neural_solver.model import PhysicsNeuralSolver, count_weights   # the network
from physics_neural_solver.train import TrainingSettings, precision_context, training_step
from physics_neural_solver.training_pool import TrainingPool        # training states

PROJECTED_STEPS = (10_000, 40_000, 100_000)                         # step counts for the time projection


def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize()                                    # wait for the GPU before reading the clock


def is_out_of_memory(error):
    return "out of memory" in str(error).lower()                    # works for every PyTorch version


def time_training(grid_name, network, batch, precision, device, warmup, timed):
    settings = TrainingSettings(grid=grid_name, network_size=network, batch_size=batch, precision=precision)
    result = {"grid": grid_name, "network": network, "batch": batch, "precision": precision}
    try:
        dt = settings.time_step_seconds / Scales().time             # time step, dimensionless
        pool = TrainingPool(max(4 * batch, 16), GRIDS[grid_name], device, dt)   # small pool (only the speed matters)
        pool.set_stage("stone")                                     # the most expensive stage
        model = PhysicsNeuralSolver(network).to(device)             # the network
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   # optimiser
        context = precision_context(precision, device)              # fp32 / tf32 / bf16
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()                    # measure memory of this run only
        times = []
        for step in range(warmup + timed):
            synchronise(device)
            start = time.perf_counter()
            training_step(model, optimizer, pool, settings, context, pool.simulation.h, dt)   # one real step
            synchronise(device)
            if step >= warmup:
                times.append(time.perf_counter() - start)           # seconds for this step
        median = statistics.median(times)                           # typical step time
        result.update({"status": "ok", "weights": count_weights(model), "seconds_per_step": round(median, 4),
                       "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None})
        for steps in PROJECTED_STEPS:
            result[f"hours_for_{steps // 1000}k_steps"] = round(steps * median / 3600, 2)   # projected training time
    except RuntimeError as error:
        result["status"] = "out of memory" if is_out_of_memory(error) else f"error: {error}"
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def time_solver_step(grid_name, network, precision, device, warmup, timed):
    result = {"grid": grid_name, "network": network, "precision": precision}
    dt = (1.0 / 60.0) / Scales().time                               # one frame
    pool = TrainingPool(1, GRIDS[grid_name], device, dt)            # one scene
    model = PhysicsNeuralSolver(network).to(device).eval()          # the network
    context = precision_context(precision, device)                  # fp32 / tf32 / bf16
    times = []
    for step in range(warmup + timed):
        _, state, box = pool.sample(1)                              # the scene
        synchronise(device)
        start = time.perf_counter()
        with context:
            model(state, box, pool.simulation.h)                    # one solver step
        synchronise(device)
        if step >= warmup:
            times.append(time.perf_counter() - start)
    median = statistics.median(times)
    result.update({"seconds_per_step": round(median, 5), "steps_per_second": round(1.0 / median, 1)})
    return result


def main():
    parser = argparse.ArgumentParser(description="Timing benchmark of the physics neural solver.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="tiny run to check the code")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":                                       # make it obvious when the GPU is not used
        print("NOTE: no GPU - this benchmark runs on the CPU, so its times say nothing about GPU training.", flush=True)

    grids = ["coarse"] if args.quick else ["coarse", "fine"]        # what to measure
    networks = ["small"] if args.quick else ["small", "medium", "large"]
    precisions = ["fp32"] if args.quick else ["fp32", "tf32", "bf16"]
    batches = {"coarse": [2] if args.quick else [8, 16, 32], "fine": [2, 4, 8]}
    warmup, timed = (1, 2) if args.quick else (3, 10)

    out_dir = Path("outputs/benchmark") / datetime.now().strftime("%Y%m%d-%H%M%S")   # where to write
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_report()                              # GPU, driver, versions
    training, solver = [], []                                       # results
    for grid in grids:
        for network in networks:
            for precision in precisions:
                solver.append(time_solver_step(grid, network, precision, device, warmup, timed))
                print(json.dumps(solver[-1]))
                for batch in batches[grid]:
                    training.append(time_training(grid, network, batch, precision, device, warmup, timed))
                    print(json.dumps(training[-1]))

    (out_dir / "results.json").write_text(json.dumps({"environment": environment, "training": training,
                                                      "solver_step": solver}, indent=2))
    lines = ["# Benchmark", "", f"GPU: {environment.get('gpu_name', 'none')}  |  torch {environment['torch']}  |  "
             f"CUDA {environment['torch_cuda_build']}", "", "## Training step", "",
             "| grid | network | batch | precision | s/step | peak GB | h for 10k / 40k / 100k steps |", "|---|---|---|---|---|---|---|"]
    for r in training:
        if r["status"] == "ok":
            hours = " / ".join(str(r[f"hours_for_{s // 1000}k_steps"]) for s in PROJECTED_STEPS)
            lines.append(f"| {r['grid']} | {r['network']} | {r['batch']} | {r['precision']} | {r['seconds_per_step']} | {r['peak_memory_gb']} | {hours} |")
        else:
            lines.append(f"| {r['grid']} | {r['network']} | {r['batch']} | {r['precision']} | {r['status']} | | |")
    lines += ["", "## One solver step (batch 1, no gradients)", "", "| grid | network | precision | s/step | steps/s |", "|---|---|---|---|---|"]
    lines += [f"| {r['grid']} | {r['network']} | {r['precision']} | {r['seconds_per_step']} | {r['steps_per_second']} |" for r in solver]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nResults in {out_dir}/summary.md")


if __name__ == "__main__":
    main()
