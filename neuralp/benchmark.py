"""
Stage 0: timing benchmark. It does NOT train the real model; it measures how fast training
and running the model would be on this machine.

For every combination of grid (coarse / fine), network size (small / medium / large),
batch size and precision (fp32 / tf32 / bf16) it runs a few real training steps
(network -> physics loss with all 8 terms -> backward -> Adam update) and records:
    time per training step, samples per second, peak GPU memory, and whether it fit in memory.
It also measures the time of one solver step (batch 1, no gradients), which is what the
browser will have to run every frame.

    python -m neuralp.benchmark                 # full run on the GPU (writes outputs/benchmark/<time>/)
    python -m neuralp.benchmark --quick --device cpu   # tiny code check, a few seconds
"""

import argparse
import contextlib
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch

from . import grid as grid_ops
from .check_environment import environment_report
from .config import GRIDS, BoxGeometry, Scales
from .network import NETWORK_PRESETS, WaterUNet3D, count_parameters
from .physics_loss import advance_state, physics_loss
from .scene import NETWORK_INPUT_CHANNELS, NETWORK_OUTPUT_CHANNELS, network_input, random_scene_batch

PROJECTED_STEP_COUNTS = (20_000, 50_000, 100_000, 200_000)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Timing benchmark for the neural water solver.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--grids", nargs="+", default=["coarse", "fine"], choices=list(GRIDS))
    parser.add_argument("--networks", nargs="+", default=list(NETWORK_PRESETS), choices=list(NETWORK_PRESETS))
    parser.add_argument("--batch-sizes-coarse", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--batch-sizes-fine", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "bf16"], choices=["fp32", "tf32", "bf16"])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--time-step", type=float, default=0.1, help="dimensionless solver time step")
    parser.add_argument("--compile", action="store_true", help="also try torch.compile (slow to start)")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--quick", action="store_true", help="tiny run to check the code works")
    args = parser.parse_args()
    if args.quick:
        args.grids, args.networks = ["coarse"], ["small"]
        args.batch_sizes_coarse, args.precisions = [2], ["fp32"]
        args.warmup_steps, args.timed_steps = 1, 2
    return args


def set_precision(precision: str, device: str):
    """fp32 = strict float32; tf32 = float32 with tensor cores; bf16 = network in bfloat16, physics in float32."""
    if device == "cuda":
        strict = precision == "fp32"
        torch.backends.cuda.matmul.allow_tf32 = not strict
        torch.backends.cudnn.allow_tf32 = not strict
    if precision == "bf16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def is_out_of_memory(error: Exception) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    return isinstance(error, oom_type) or "out of memory" in str(error).lower()


def self_tests(device) -> dict:
    """Quick correctness checks before timing anything."""
    results = {}
    scene = random_scene_batch(2, GRIDS["coarse"], device)
    velocity = grid_ops.velocity_from_potential(scene.potential, scene.cell_size)
    divergence = grid_ops.divergence_of_face_velocity(velocity, scene.cell_size)[:, :, 2:-2, 2:-2, 2:-2]
    results["max_divergence_of_curl_A"] = float(divergence.abs().max())
    results["max_velocity"] = float(velocity.abs().max())
    model = WaterUNet3D(NETWORK_INPUT_CHANNELS, NETWORK_OUTPUT_CHANNELS, base_channels=8).to(device)
    output = model(network_input(scene))
    results["network_output_shape_ok"] = tuple(output.shape) == (2, NETWORK_OUTPUT_CHANNELS, *scene.phi.shape[2:])
    total, terms = physics_loss(scene, advance_state(scene, output), time_step=0.1)
    results["loss_is_finite"] = bool(torch.isfinite(total))
    results["loss_terms_at_start"] = {name: float(value.detach()) for name, value in terms.items()}
    return results


def time_training(grid_name, network_name, batch_size, precision, args) -> dict:
    device = args.device
    record = {"kind": "training", "grid": grid_name, "network": network_name, "batch_size": batch_size,
              "precision": precision, "compiled": args.compile}
    grid = GRIDS[grid_name]
    record["cells"] = "x".join(str(n) for n in grid.total_cells(BoxGeometry()))
    model = WaterUNet3D(NETWORK_INPUT_CHANNELS, NETWORK_OUTPUT_CHANNELS, base_channels=NETWORK_PRESETS[network_name]).to(device)
    record["parameters"] = count_parameters(model)
    runnable = torch.compile(model) if args.compile else model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    autocast = set_precision(precision, device)
    try:
        scene = random_scene_batch(batch_size, grid, device)
        inputs = network_input(scene)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        step_times, last_loss = [], None
        for step in range(args.warmup_steps + args.timed_steps):
            synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                output = runnable(inputs)
            total, _ = physics_loss(scene, advance_state(scene, output), args.time_step)
            total.backward()
            optimizer.step()
            synchronize(device)
            if step >= args.warmup_steps:
                step_times.append(time.perf_counter() - start)
            last_loss = float(total.detach())
        median = statistics.median(step_times)
        record.update({
            "status": "ok",
            "seconds_per_step_median": round(median, 5),
            "seconds_per_step_mean": round(statistics.mean(step_times), 5),
            "samples_per_second": round(batch_size / median, 2),
            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            "loss_finite": bool(torch.isfinite(torch.tensor(last_loss))),
        })
        for steps in PROJECTED_STEP_COUNTS:
            record[f"hours_for_{steps // 1000}k_steps"] = round(steps * median / 3600, 2)
    except Exception as error:  # noqa: BLE001 - record and continue with the next combination
        record["status"] = "out_of_memory" if is_out_of_memory(error) else f"error: {type(error).__name__}: {error}"
    finally:
        del model, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()
    return record


def time_inference(grid_name, network_name, precision, args) -> dict:
    """One solver step at batch 1 without gradients: network -> next state -> velocity."""
    device = args.device
    record = {"kind": "inference", "grid": grid_name, "network": network_name, "batch_size": 1,
              "precision": precision, "compiled": False}
    grid = GRIDS[grid_name]
    record["cells"] = "x".join(str(n) for n in grid.total_cells(BoxGeometry()))
    model = WaterUNet3D(NETWORK_INPUT_CHANNELS, NETWORK_OUTPUT_CHANNELS, base_channels=NETWORK_PRESETS[network_name]).to(device).eval()
    record["parameters"] = count_parameters(model)
    autocast = set_precision(precision, device)
    try:
        scene = random_scene_batch(1, grid, device)
        inputs = network_input(scene)
        times = []
        with torch.no_grad():
            for step in range(args.warmup_steps + args.timed_steps):
                synchronize(device)
                start = time.perf_counter()
                with autocast:
                    output = model(inputs)
                next_state = advance_state(scene, output)
                grid_ops.velocity_from_potential(next_state.potential, scene.cell_size)
                synchronize(device)
                if step >= args.warmup_steps:
                    times.append(time.perf_counter() - start)
        median = statistics.median(times)
        record.update({"status": "ok", "seconds_per_step_median": round(median, 5),
                       "solver_steps_per_second": round(1.0 / median, 1)})
    except Exception as error:  # noqa: BLE001
        record["status"] = "out_of_memory" if is_out_of_memory(error) else f"error: {type(error).__name__}: {error}"
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return record


def write_summary(path: Path, environment: dict, tests: dict, records: list):
    lines = ["# Benchmark summary", ""]
    lines += [f"- GPU: {environment.get('gpu_name', 'none')} ({environment.get('gpu_memory_gb', '?')} GB), "
              f"torch {environment['torch']} (CUDA {environment['torch_cuda_build']})",
              f"- nvidia-smi: {environment['nvidia_smi']}",
              f"- Self-tests: max div(curl A) = {tests['max_divergence_of_curl_A']:.2e} "
              f"(should be ~0), loss finite = {tests['loss_is_finite']}", ""]
    if environment["warnings"]:
        lines += ["**Warnings:** " + " / ".join(environment["warnings"]), ""]

    lines += ["## Training (one step = network + physics loss + backward + Adam)", "",
              "| grid | network | params | batch | precision | s/step | samples/s | peak GB | h per 20k / 50k / 100k / 200k steps |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        if r["kind"] != "training":
            continue
        if r["status"] != "ok":
            lines.append(f"| {r['grid']} | {r['network']} | {r.get('parameters', '')} | {r['batch_size']} | "
                         f"{r['precision']} | {r['status']} | | | |")
            continue
        hours = " / ".join(str(r[f"hours_for_{s // 1000}k_steps"]) for s in PROJECTED_STEP_COUNTS)
        lines.append(f"| {r['grid']} | {r['network']} | {r['parameters']:,} | {r['batch_size']} | {r['precision']} | "
                     f"{r['seconds_per_step_median']} | {r['samples_per_second']} | {r['peak_memory_gb']} | {hours} |")

    lines += ["", "## One solver step (batch 1, no gradients): what the viewer runs per step", "",
              "| grid | network | precision | s/step | steps/s |", "|---|---|---|---|---|"]
    for r in records:
        if r["kind"] == "inference":
            if r["status"] == "ok":
                lines.append(f"| {r['grid']} | {r['network']} | {r['precision']} | "
                             f"{r['seconds_per_step_median']} | {r['solver_steps_per_second']} |")
            else:
                lines.append(f"| {r['grid']} | {r['network']} | {r['precision']} | {r['status']} | |")
    lines += ["", "Note: the hours assume a number of training steps; how many steps are really needed",
              "will only be known from the first coarse training runs. GPU timings here are on the cluster GPU;",
              "the browser (WebGPU on a laptop) will be slower."]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_arguments()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible. Run `python -m neuralp.check_environment` to see why.")

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_report()
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    print(json.dumps(environment, indent=2))

    tests = self_tests(args.device)
    (run_dir / "self_tests.json").write_text(json.dumps(tests, indent=2))
    print("self-tests:", json.dumps(tests, indent=2))

    records = []
    csv_path = run_dir / "results.csv"
    fieldnames = None

    def save(record):
        nonlocal fieldnames
        records.append(record)
        print(json.dumps(record))
        if fieldnames is None:
            fieldnames = sorted({key for r in records for key in r} | {f"hours_for_{s // 1000}k_steps" for s in PROJECTED_STEP_COUNTS}
                                | {"seconds_per_step_mean", "samples_per_second", "peak_memory_gb", "loss_finite",
                                   "solver_steps_per_second", "seconds_per_step_median", "status"})
            with csv_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        with csv_path.open("a", newline="") as f:  # written as we go, so a cut-off job still leaves results
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(record)

    for grid_name in args.grids:
        batch_sizes = args.batch_sizes_coarse if grid_name == "coarse" else args.batch_sizes_fine
        for network_name in args.networks:
            for precision in args.precisions:
                save(time_inference(grid_name, network_name, precision, args))
                for batch_size in batch_sizes:
                    save(time_training(grid_name, network_name, batch_size, precision, args))

    (run_dir / "results.json").write_text(json.dumps(records, indent=2))
    write_summary(run_dir / "summary.md", environment, tests, records)
    print(f"\nDone. Results in {run_dir}/ (summary.md is the easiest to read)")


if __name__ == "__main__":
    main()
