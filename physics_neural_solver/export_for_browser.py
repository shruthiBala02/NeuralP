"""
export_for_browser.py  -  write a trained network in a form the Three.js viewer can load

    python -m physics_neural_solver.export_for_browser --run coarse_small

Writes outputs/export/<run>/:
    weights.bin     all weights, float32, one after another
    model.json      the network layout (every layer: name, shape, where its numbers are in weights.bin) plus
                    everything the viewer needs to use it: grid, cell size, time step, reference scales, input order
The viewer runs the network itself with WebGPU compute shaders (3D convolutions), reading these two files.
"""

import argparse                                                     # command-line options
import json                                                         # the layout file
from pathlib import Path                                            # file paths

import numpy as np                                                  # writing the weights
import torch                                                        # loading the checkpoint

from config import GRIDS, Scales                                    # grids and reference scales
from physics_neural_solver.model import INPUT_CHANNELS, NETWORK_SIZES, OUTPUT_CHANNELS, PhysicsNeuralSolver


def main():
    parser = argparse.ArgumentParser(description="Export a trained network for the browser viewer.")
    parser.add_argument("--run", default="coarse_small", help="run name under outputs/runs/")
    args = parser.parse_args()

    checkpoint = torch.load(Path("outputs/runs") / args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]                               # how it was trained
    model = PhysicsNeuralSolver(settings["network_size"])           # same architecture
    model.load_state_dict(checkpoint["model"])                      # trained weights
    scales, grid = Scales(), GRIDS[settings["grid"]]                # reference scales and grid

    layers, offset, chunks = [], 0, []                              # layout, position in the file, the numbers
    for name, tensor in model.state_dict().items():                 # every weight tensor, in order
        values = tensor.detach().float().cpu().numpy().ravel()      # flatten to float32
        layers.append({"name": name, "shape": list(tensor.shape), "offset": offset, "count": int(values.size)})
        chunks.append(values)                                       # keep the numbers
        offset += int(values.size)                                  # next layer starts after this one

    out_dir = Path("outputs/export") / args.run                     # where to write
    out_dir.mkdir(parents=True, exist_ok=True)
    np.concatenate(chunks).astype(np.float32).tofile(out_dir / "weights.bin")   # all weights in one file
    manifest = {
        "network": {"size": settings["network_size"], "base_channels": NETWORK_SIZES[settings["network_size"]],
                    "depth": model.network.depth, "input_channels": INPUT_CHANNELS, "output_channels": OUTPUT_CHANNELS,
                    "normalisation": "GroupNorm(min(8, channels))", "activation": "SiLU"},
        "input_order": ["phi", "A_x", "A_y", "A_z", "p", "wall_mask", "stone_mask", "stone_vx", "stone_vy", "stone_vz",
                        "gravity_x", "gravity_y", "gravity_z", "tilt_rate", "tilt_acceleration"],
        "output_order": ["dA_x", "dA_y", "dA_z", "dp", "dphi"],
        "grid": {"name": grid.name, "cells_with_walls": list(grid.total_cells(scales.box)), "cell_size_m": grid.cell_size},
        "time_step_s": settings["time_step_seconds"],
        "scales": {"length_m": scales.length, "speed_m_s": scales.speed, "time_s": scales.time,
                   "pressure_pa": scales.water.density * scales.water.gravity * scales.length},
        "trained_steps": checkpoint["step"],
        "layers": layers,
    }
    (out_dir / "model.json").write_text(json.dumps(manifest, indent=2))   # the layout
    print(f"Wrote {out_dir}/weights.bin ({offset:,} weights, {offset * 4 / 1e6:.1f} MB) and model.json")


if __name__ == "__main__":
    main()
