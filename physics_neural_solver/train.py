"""
train.py  -  train the physics neural solver (physics loss only, no simulation data)

    python -m physics_neural_solver.train                         # coarse grid, small network (defaults below)
    python -m physics_neural_solver.train --smoke-test            # 30 steps on a tiny pool: checks that everything runs
    python -m physics_neural_solver.train --resume                # continue from the last checkpoint of this run

Each training step:
    1. take a random batch of scenes from the training pool
    2. the network predicts the water one step later
    3. the physics loss checks the prediction against the equations (physical_equations.loss_terms)
    4. update the network weights (Adam)
    5. put the prediction back into the pool; move tilt and stones exactly; random resets and user actions

Output (outputs/runs/<run name>/):
    log.csv          one line every `log_every` steps: every loss term, learning rate, speed
    losses.png       graph of the total loss and each equation's loss (refreshed at every checkpoint)
    checkpoint.pt    network + optimiser + progress (written every `checkpoint_every` steps and at the end)
    checkpoint_end_of_calm.pt, checkpoint_end_of_tilt.pt   the network at the end of each stage (evaluate them with
                     --checkpoint, e.g. to check sloshing before stones were added)
    settings.json    the settings of this run
"""

import argparse                                                     # command-line options
import csv                                                          # the log file
import json                                                         # the settings file
import math                                                         # cosine learning-rate schedule
import time                                                         # timing
from contextlib import nullcontext                                  # "no autocast"
from dataclasses import asdict, dataclass, field                    # the settings container
from pathlib import Path                                            # file paths

import torch                                                        # tensors and training

from config import GRIDS, Scales                                    # grids and reference scales
from physics_neural_solver import physical_equations as eq          # all equations and the loss
from physics_neural_solver.model import PhysicsNeuralSolver, count_weights   # the network
from physics_neural_solver.plot_losses import plot as plot_losses  # the loss graph (losses.png)
from physics_neural_solver.terminal_log import TerminalLog          # progress in the terminal
from physics_neural_solver.training_pool import TrainingPool        # where the training states come from


# ------------------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------------------
@dataclass
class TrainingSettings:
    run_name: str = "coarse_small"                                  # folder name under outputs/runs/
    grid: str = "coarse"                                            # "coarse" (2.5 cm) or "fine" (1.25 cm)
    network_size: str = "small"                                     # "small", "medium" or "large"
    batch_size: int = 16                                            # scenes per training step
    pool_size: int = 256                                            # scenes in the training pool
    time_step_seconds: float = 1.0 / 60.0                           # one network step = one frame at 60 fps
    stage_steps: dict = field(default_factory=lambda: {"calm": 3000, "tilt": 17000, "stone": 20000})   # curriculum (tilt longer: it was still improving at 12k)
    learning_rate: float = 5e-4                                     # Adam learning rate at the start
    final_learning_rate: float = 2e-5                               # ... at the end (cosine decay)
    gradient_clip: float = 1.0                                      # largest allowed gradient size (stability)
    importance: dict = field(default_factory=lambda: {              # how much each equation matters after balancing
        "momentum (eq. 2)": 1.0, "level set (eq. 5)": 1.0, "eikonal (eq. 6)": 0.1, "volume (eq. 7)": 100.0,   # volume x100:
        # its residual is a relative error, squared - at x1, losing 3 % of the water cost only 0.0009 and the water drained
        # away (run v2). At x100, 1 % lost costs ~0.01, as much as the other equations.
        "no shear (eq. 9)": 0.1, "no penetration (eq. 10)": 20.0, "wall law (eq. 11)": 1.0,
        "stone boundary (eq. 13)": 1.0})                            # fixed weights; walls x20 as in Wandel et al. Stone x1:
                                                                    # at x20 it swamped all water equations (first run)
    stone_fade_in_steps: int = 3000                                 # the stone weight grows from 0 to its value over this many steps
    first_throw_speed_m_s: float = 1.0                              # throws start slow (a 3 m/s stone moves 2 cells per step) ...
    full_throw_speed_after: float = 0.5                             # ... and reach full speed after this share of the stone stage
    precision: str = "tf32"                                         # "fp32", "tf32" or "bf16" (bf16: network only)
    log_every: int = 50                                             # steps between log lines
    checkpoint_every: int = 1000                                    # steps between checkpoints
    seed: int = 0                                                   # random seed
    device: str = "auto"                                            # "auto", "cuda", "mps" or "cpu"

    @property
    def total_steps(self) -> int:
        return sum(self.stage_steps.values())                       # all stages together


# ------------------------------------------------------------------------------------------------------
# Loss: fixed weights
# ------------------------------------------------------------------------------------------------------
def importance_at(step: int, settings: TrainingSettings) -> dict:
    """The weights at this step: the stone weight fades in at the start of the stone stage, so the first
    (large) water-stone errors don't swamp the water equations."""
    importance = dict(settings.importance)                          # a copy of the fixed weights
    stone_start = settings.stage_steps["calm"] + settings.stage_steps["tilt"]   # first step of the stone stage
    fade = min(max((step - stone_start) / max(settings.stone_fade_in_steps, 1), 0.0), 1.0)   # 0 -> 1
    importance["stone boundary (eq. 13)"] *= fade                   # (0 before the stone stage: there are no stones)
    return importance


def throw_speed_at(step: int, settings: TrainingSettings, scales: Scales) -> float:
    """Largest throw speed (dimensionless) at this step: from `first_throw_speed_m_s` up to the slider maximum
    (3 m/s), reached after `full_throw_speed_after` of the stone stage."""
    stone_start = settings.stage_steps["calm"] + settings.stage_steps["tilt"]   # first step of the stone stage
    ramp_steps = max(settings.full_throw_speed_after * settings.stage_steps["stone"], 1)   # length of the speed-up
    progress = min(max((step - stone_start) / ramp_steps, 0.0), 1.0)   # 0 -> 1
    first, full = settings.first_throw_speed_m_s, scales.box.max_throw_speed   # m/s
    return (first + (full - first) * progress) / scales.speed       # m/s -> dimensionless


def weighted_total(terms: dict, importance: dict) -> torch.Tensor:
    """total = sum over the 8 equations of  importance_i * L_i.
    Fixed weights, as in Wandel et al. (2021). The equations are dimensionless, so every term is naturally of
    order 1. (Adaptive weights - dividing each term by its own running size - were tested and BLOCKED learning:
    terms that start near zero get huge gradients and swamp the training signal. See tests/test_training.py.)"""
    return sum(importance[name] * value for name, value in terms.items())   # weighted sum


# ------------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------------
def pick_device(choice: str) -> torch.device:
    """The GPU, always - unless the CPU is chosen on purpose with --device cpu.
    Training on the CPU by accident would be extremely slow, so with "auto" and no GPU visible we stop instead."""
    if choice != "auto":                                            # chosen on purpose
        return torch.device(choice)
    if torch.cuda.is_available():                                   # NVIDIA GPU (the cluster)
        torch.backends.cudnn.benchmark = True                       # let cuDNN pick the fastest 3D convolutions
        return torch.device("cuda")
    raise SystemExit("No GPU is visible to PyTorch, so training will NOT start (it would be far too slow on the CPU).\n"
                     "Check:  nvidia-smi   and   python check_environment.py   (are you on a GPU node? is the\n"
                     "environment activated?). To train on the CPU anyway, add  --device cpu")


def precision_context(precision: str, device: torch.device):
    if device.type == "cuda":                                       # tensor-core settings only exist on NVIDIA GPUs
        strict = precision == "fp32"                                # strict float32?
        torch.backends.cuda.matmul.allow_tf32 = not strict          # tf32 for matrix products
        torch.backends.cudnn.allow_tf32 = not strict                # tf32 for convolutions
    if precision == "bf16":                                         # network in bfloat16 (the physics stays float32)
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()                                            # full float32 network


def stage_at(step: int, stage_steps: dict) -> str:
    end = 0                                                         # where the current stage ends
    for stage, steps in stage_steps.items():                        # calm, tilt, stone (in order)
        end += steps
        if step < end:
            return stage
    return list(stage_steps)[-1]                                    # after the end: the last stage


def learning_rate_at(step: int, settings: TrainingSettings) -> float:
    progress = min(step / max(settings.total_steps, 1), 1.0)        # 0 at the start, 1 at the end
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))             # 1 -> 0 smoothly
    return settings.final_learning_rate + (settings.learning_rate - settings.final_learning_rate) * cosine


def training_step(model, optimizer, pool, settings, precision, h, dt, importance=None):
    """One training step. Returns (total loss, dict of the 8 raw loss terms, number of resets).
    importance: the weights for this step (default: the fixed settings.importance)."""
    indices, state, box = pool.sample(settings.batch_size)          # 1. random batch from the pool
    with precision:                                                 # (bf16 only inside the network)
        next_state = model(state, box, h)                           # 2. the network's prediction
    terms = eq.loss_terms(state, next_state, box, pool.simulation.positions,
                          pool.simulation.initial_volume[indices], h, dt)   # 3. the physics loss
    total = weighted_total(terms, importance or settings.importance)   # weighted sum of the 8 equations
    optimizer.zero_grad(set_to_none=True)                           # 4. clear old gradients
    total.backward()                                                #    gradients of the loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)   # avoid huge updates
    optimizer.step()                                                #    update the weights
    resets = pool.put_back(indices, next_state)                     # 5. prediction back into the pool
    return total.detach(), {name: value.detach() for name, value in terms.items()}, resets


# ------------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------------
def train(settings: TrainingSettings, resume: bool = False):
    torch.manual_seed(settings.seed)                                # reproducible network initialisation
    device = pick_device(settings.device)                           # where to train
    scales = Scales()                                               # reference scales
    grid = GRIDS[settings.grid]                                     # coarse or fine grid
    dt = settings.time_step_seconds / scales.time                   # time step, dimensionless
    run_dir = Path("outputs/runs") / settings.run_name              # where everything is written
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps(asdict(settings), indent=2))   # remember the settings

    model = PhysicsNeuralSolver(settings.network_size).to(device)   # the network
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)   # Adam optimiser
    pool = TrainingPool(settings.pool_size, grid, device, dt, scales, settings.seed)   # training states
    h = pool.simulation.h                                           # cell size, dimensionless
    precision = precision_context(settings.precision, device)       # fp32 / tf32 / bf16

    start_step = 0                                                  # where training starts
    checkpoint_path = run_dir / "checkpoint.pt"
    if resume and checkpoint_path.exists():                         # continue an earlier run
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])                  # network weights
        optimizer.load_state_dict(checkpoint["optimizer"])          # optimiser memory
        start_step = checkpoint["step"]                             # progress

    terminal = TerminalLog(settings.total_steps, start_step, gpu=device.type == "cuda")   # progress in the terminal
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "none (CPU)"
    terminal.settings_block({
        "run": f"{settings.run_name}   ->   {run_dir}/",
        "device": f"{device}   GPU: {gpu_name}",
        "grid": f"{settings.grid}, {tuple(pool.simulation.positions.shape[2:])} cells (with walls)",
        "network": f"{settings.network_size}, {count_weights(model):,} weights, precision {settings.precision}",
        "batch / pool": f"{settings.batch_size} scenes per step / {settings.pool_size} scenes in the pool",
        "time step": f"{settings.time_step_seconds * 1000:.1f} ms of water time per network step",
        "stages (steps)": ", ".join(f"{name} {steps:,}" for name, steps in settings.stage_steps.items())
                          + f"   (total {settings.total_steps:,})",
        "stone stage": f"stone weight fades in over {settings.stone_fade_in_steps:,} steps; throws "
                       f"{settings.first_throw_speed_m_s:.1f} -> {scales.box.max_throw_speed:.1f} m/s",
        "pool health": "reset when volume is off > 5 % or water > 6 m/s; 1/4 of the scenes never get a stone",
        "learning rate": f"{settings.learning_rate:.1e} -> {settings.final_learning_rate:.1e} (cosine)",
        "log / checkpoint": f"every {settings.log_every} / {settings.checkpoint_every} steps",
        "start": f"resumed from step {start_step:,} (the pool starts again from calm water)" if start_step else "step 0",
    })
    terminal.say("Training has no epochs (no dataset): progress is in steps. Columns: loss of every equation (eq. number).")

    log_path = run_dir / "log.csv"                                  # the log file
    new_log = not (resume and log_path.exists())                    # start a new log, or append
    log_file = log_path.open("w" if new_log else "a", newline="")
    log = csv.writer(log_file)
    term_names = list(settings.importance)                          # the 8 equations, in order
    if new_log:
        log.writerow(["step", "stage", "total"] + term_names + ["learning_rate", "seconds_per_step",
                                                                "resets", "volume_error_mean", "potential_max", "stones_active",
                                                                "scene_age_mean_s", "scene_age_max_s"])

    clock, resets_since_log = time.time(), 0                        # for the speed column
    for step in range(start_step, settings.total_steps):
        stage = stage_at(step, settings.stage_steps)                # calm, tilt or stone
        if stage != pool.stage:                                     # a new curriculum stage begins
            if step > start_step:                                   # keep the network as it was at the end of the old stage
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                            "settings": asdict(settings)}, run_dir / f"checkpoint_end_of_{pool.stage}.pt")
                terminal.say(f"saved checkpoint_end_of_{pool.stage}.pt (the network after stage '{pool.stage}')")
            pool.set_stage(stage)
            terminal.say(f"---- step {step:,}: stage '{stage}' begins "
                         f"({ {'calm': 'water at rest', 'tilt': 'the box tilts', 'stone': 'stones are thrown in'}[stage] }) ----")
        for group in optimizer.param_groups:                        # learning-rate schedule
            group["lr"] = learning_rate_at(step, settings)

        pool.max_throw_speed = throw_speed_at(step, settings, scales)   # slow throws first
        total, terms, resets = training_step(model, optimizer, pool, settings, precision, h, dt,
                                             importance_at(step, settings))
        resets_since_log += resets

        if (step + 1) % settings.log_every == 0 or step + 1 == settings.total_steps:   # write a log line
            seconds = (time.time() - clock) / settings.log_every    # average time per step
            stats = pool.statistics()                               # health of the pool
            log.writerow([step + 1, stage, f"{float(total):.4e}"] + [f"{float(terms[n]):.4e}" for n in term_names]
                         + [f"{optimizer.param_groups[0]['lr']:.2e}", f"{seconds:.3f}", resets_since_log,
                            f"{stats['volume_error_mean']:.4f}", f"{stats['potential_max']:.3f}", stats["stones_active"],
                            f"{stats['scene_age_mean_s']:.2f}", f"{stats['scene_age_max_s']:.2f}"])
            log_file.flush()                                        # keep the file up to date
            gpu_memory = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
            terminal.row(step + 1, stage, float(total), terms, optimizer.param_groups[0]["lr"], seconds,
                         resets_since_log, gpu_memory)
            clock, resets_since_log = time.time(), 0

        if (step + 1) % settings.checkpoint_every == 0 or step + 1 == settings.total_steps:   # save progress
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step + 1,
                        "settings": asdict(settings)}, checkpoint_path)
            try:
                plot_losses(run_dir)                                # refresh losses.png
                terminal.say(f"checkpoint saved at step {step + 1:,}  ->  checkpoint.pt, losses.png")
            except Exception as error:                              # a plotting problem must never stop training
                terminal.say(f"checkpoint saved at step {step + 1:,} (could not draw losses.png: {error})")

    log_file.close()                                                # done
    terminal.say(f"Finished {settings.total_steps:,} steps.  Checkpoint: {checkpoint_path}   Log: {log_path}   "
                 f"Graphs: {run_dir}/losses.png")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the physics neural solver (physics loss only).")
    parser.add_argument("--run-name", default=None, help="folder name under outputs/runs/")
    parser.add_argument("--grid", choices=list(GRIDS), default="coarse")
    parser.add_argument("--network", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=None, help="total steps (the three stages keep their proportions)")
    parser.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="tf32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="continue from this run's checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="30 steps on a tiny pool, just to check the code")
    args = parser.parse_args()

    settings = TrainingSettings(grid=args.grid, network_size=args.network, batch_size=args.batch_size,
                                precision=args.precision, device=args.device)
    settings.run_name = args.run_name or f"{args.grid}_{args.network}"
    if args.steps:                                                  # scale the stages to the requested total
        scale = args.steps / settings.total_steps
        settings.stage_steps = {name: max(1, round(steps * scale)) for name, steps in settings.stage_steps.items()}
    if args.smoke_test:                                             # tiny run
        settings.run_name, settings.pool_size, settings.batch_size = "smoke_test", 8, 2
        settings.stage_steps = {"calm": 10, "tilt": 10, "stone": 10}
        settings.log_every, settings.checkpoint_every = 5, 30
    train(settings, resume=args.resume)


if __name__ == "__main__":
    main()
