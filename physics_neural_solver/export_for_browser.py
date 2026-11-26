"""
export_for_browser.py  -  write a trained network in a form the Three.js viewer can load

    python -m physics_neural_solver.export_for_browser --run coarse_small     # a trained run
    python -m physics_neural_solver.export_for_browser --test-network         # random weights, only to test the viewer

Writes outputs/export/<name>/:
    weights.bin     all weights, float32, one after another
    model.json      the network layout (every layer: name, shape, where its numbers are in weights.bin) plus what the
                    viewer needs: grid, cell size, time step, reference scales, input order, and the test case below
    test_case.bin   one scene (water, stone, tilt) and the network's exact output for it, computed by PyTorch.
                    The viewer runs the same step on the GPU and compares (?solver=neural&test=equivalence):
                    if the numbers match, the browser runs the network exactly like PyTorch.
All arrays are float32 in PyTorch's order (x slowest, z fastest) and dimensionless.
"""

import argparse                                                     # command-line options
import json                                                         # the layout file
import math                                                         # radians
from pathlib import Path                                            # file paths

import numpy as np                                                  # writing binary files
import torch                                                        # loading the checkpoint, the test case

from config import GRIDS, Scales                                    # grids and reference scales
from physics_neural_solver import physical_equations as eq          # velocity from the potential
from physics_neural_solver.model import INPUT_CHANNELS, NETWORK_SIZES, OUTPUT_CHANNELS, PhysicsNeuralSolver
from physics_neural_solver.simulation import Simulation             # to build the test scene


def load_network(args):
    """The trained network (or a random one for --test-network) and its settings."""
    if args.test_network:                                           # random weights, also in the last layer
        torch.manual_seed(1)
        model = PhysicsNeuralSolver("small")
        torch.nn.init.normal_(model.network.output.weight, std=0.05)   # make the output non-zero
        torch.nn.init.normal_(model.network.output.bias, std=0.01)
        settings = {"network_size": "small", "grid": args.grid, "time_step_seconds": 1.0 / 60.0}
        return model, settings, 0, f"test_network_{args.grid}"
    checkpoint = torch.load(Path("outputs/runs") / args.run / args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]                               # how it was trained
    model = PhysicsNeuralSolver(settings["network_size"])           # same architecture
    model.load_state_dict(checkpoint["model"])                      # trained weights
    name = args.run if args.checkpoint == "checkpoint.pt" else f"{args.run}_{Path(args.checkpoint).stem.replace('checkpoint_', '')}"
    return model, settings, checkpoint["step"], name                # e.g. coarse_small or coarse_small_end_of_tilt


def network_stages(model, state, box):
    """The padded network input and the raw (padded, uncropped) network output, exactly as WaterUNet3D computes them."""
    from physics_neural_solver.model import network_input        # the 15 input channels
    import torch.nn.functional as F                                 # padding and pooling
    net = model.network                                             # the U-Net
    x = network_input(state, box)
    multiple = 2 ** net.depth                                       # pad to a multiple of 2^depth (same as forward)
    padding = []
    for n in reversed(x.shape[2:]):
        padding += [0, (-n) % multiple]
    x = F.pad(x, padding, mode="replicate")
    padded_input = x.clone()
    skips = []
    for block in net.down_blocks:                                   # the same steps as WaterUNet3D.forward
        x = block(x)
        skips.append(x)
        x = F.max_pool3d(x, kernel_size=2)
    x = net.bottom(x)
    for upsample, block, skip in zip(net.up_samplers, net.up_blocks, reversed(skips)):
        x = block(torch.cat([upsample(x), skip], dim=1))
    return padded_input, net.output(x)                              # (1, 15, P...) and (1, 5, P...)


def make_test_case(model, settings):
    """A scene with everything switched on: tilted moving water, a moving spinning stone, a turning box."""
    scales = Scales()                                               # reference scales
    dt = settings["time_step_seconds"] / scales.time                # time step, dimensionless
    sim = Simulation(1, GRIDS[settings["grid"]], "cpu", dt, scales) # one scene
    index = torch.tensor([0])
    sim.reset(index, torch.tensor([1.1]), torch.tensor([math.radians(6.0)]))   # 27.5 cm of water, box tilted 6 degrees
    torch.manual_seed(2)
    sim.water.potential += 0.02 * torch.randn_like(sim.water.potential)   # some motion
    sim.tilt_rate[0], sim.tilt_acceleration[0] = 0.05, -0.3         # the box is turning
    sim.throw_stone(0, (0.4, -0.3, 0.1), (0.5, -1.0, 0.2), 0.16, 2.5)   # a stone under water, moving
    sim.stones.spin[0] = torch.tensor([0.0, 0.0, 3.0])              # and spinning
    state, box = sim.inputs(index)                                  # what the network reads
    with torch.no_grad():
        next_state = model(state, box, sim.h)                       # PyTorch's answer
        velocity = eq.velocity_from_vector_potential(next_state.potential, sim.h)   # u = curl(A) on the faces
        padded_input, raw_output = network_stages(model, state, box) # two intermediate results, for debugging
    arrays = [state.phi, state.potential, state.pressure,           # input
              next_state.phi, next_state.potential, next_state.pressure, velocity,   # expected output
              padded_input, raw_output]                             # expected intermediate results
    data = np.concatenate([a[0].numpy().astype(np.float32).ravel() for a in arrays])   # one scene, PyTorch order
    s = sim.stones                                                  # the stone, for the viewer
    description = {"cells": list(state.phi.shape[2:]), "cell_size": sim.h,
                   "order": ["phi", "A(3)", "p", "expected_phi", "expected_A(3)", "expected_p", "expected_u(3)",
                             "padded_input(15, padded grid)", "raw_output(5, padded grid)"],
                   "tilt": float(sim.tilt[0]), "tilt_rate": float(sim.tilt_rate[0]),
                   "tilt_acceleration": float(sim.tilt_acceleration[0]),
                   "stone": {"active": True, "centre": s.centre[0].tolist(), "velocity": s.velocity[0].tolist(),
                             "spin": s.spin[0].tolist(), "radius": float(s.radius[0])}}
    return data, description


def main():
    parser = argparse.ArgumentParser(description="Export a trained network for the browser viewer.")
    parser.add_argument("--run", default="coarse_small", help="run name under outputs/runs/")
    parser.add_argument("--checkpoint", default="checkpoint.pt", help="e.g. checkpoint_end_of_tilt.pt (default: the final one)")
    parser.add_argument("--test-network", action="store_true", help="export a random network (to test the viewer)")
    parser.add_argument("--grid", choices=list(GRIDS), default="coarse", help="grid of the test network")
    args = parser.parse_args()

    model, settings, steps, name = load_network(args)               # the network to export
    model.eval()
    scales, grid = Scales(), GRIDS[settings["grid"]]                # reference scales and grid

    layers, offset, chunks = [], 0, []                              # layout, position in the file, the numbers
    for layer_name, tensor in model.state_dict().items():           # every weight tensor, in order
        values = tensor.detach().float().cpu().numpy().ravel()      # flatten to float32
        layers.append({"name": layer_name, "shape": list(tensor.shape), "offset": offset, "count": int(values.size)})
        chunks.append(values)                                       # keep the numbers
        offset += int(values.size)                                  # next layer starts after this one

    out_dir = Path("outputs/export") / name                         # where to write
    out_dir.mkdir(parents=True, exist_ok=True)
    np.concatenate(chunks).astype(np.float32).tofile(out_dir / "weights.bin")   # all weights in one file
    test_data, test_description = make_test_case(model, settings)   # the equivalence test
    test_data.tofile(out_dir / "test_case.bin")
    manifest = {
        "network": {"size": settings["network_size"], "base_channels": NETWORK_SIZES[settings["network_size"]],
                    "depth": model.network.depth, "input_channels": INPUT_CHANNELS, "output_channels": OUTPUT_CHANNELS,
                    "group_norm_groups": "min(8, channels)", "group_norm_eps": 1e-5, "activation": "SiLU"},
        "input_order": ["phi", "A_x", "A_y", "A_z", "p", "wall_mask", "stone_mask", "stone_vx", "stone_vy", "stone_vz",
                        "gravity_x", "gravity_y", "gravity_z", "tilt_rate", "tilt_acceleration"],
        "output_order": ["dA_x", "dA_y", "dA_z", "dp", "dphi"],
        "grid": {"name": grid.name, "cells_with_walls": list(grid.total_cells(scales.box)), "cell_size_m": grid.cell_size},
        "time_step_s": settings["time_step_seconds"],
        "scales": {"length_m": scales.length, "speed_m_s": scales.speed, "time_s": scales.time,
                   "pressure_pa": scales.water.density * scales.water.gravity * scales.length},
        "trained_steps": steps,
        "layers": layers,
        "test_case": test_description,
    }
    (out_dir / "model.json").write_text(json.dumps(manifest, indent=2))   # the layout
    print(f"Wrote {out_dir}/: weights.bin ({offset:,} weights, {offset * 4 / 1e6:.1f} MB), model.json, test_case.bin")


if __name__ == "__main__":
    main()
