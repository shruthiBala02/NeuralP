"""
model.py  -  the physics neural solver network

One 3D U-Net that steps the water forward in time:
    water now (A, p, phi) + walls + stone + tilt   ->   water one time step later (A, p, phi)
The network predicts the FLOW (change of A) and the pressure (change of p). Strict rules are then applied exactly,
so they hold whatever the network outputs (hard constraints, not things to learn):
    eq. 1   incompressible:          u = curl A
    eq. 10  no flow through walls:   A's in-wall components are 0 on the wall planes
    eq. 5   surface moves with the water: phi is carried by the flow (semi-Lagrangian), not predicted
    eq. 7   volume constant:         the surface is shifted to restore the starting volume exactly
    eq. 8   no pressure in the air
(eq. 6, phi a true distance, is restored every few steps by the simulation: simulation.py)
Pressure = still-water part + moving part. The still-water part rho g (depth) is computed exactly from the surface
(depth = phi, |g| = 1 dimensionless); the network only predicts the (small) moving part. Water at rest is therefore
exactly at rest whatever the surface does, and the network spends its effort on the flow.

Inspired by:
    U-Net             Ronneberger, Fischer & Brox, MICCAI 2015
    3D U-Net          Cicek et al., MICCAI 2016
    physics-only CNN  Wandel, Weinmann & Klein, ICLR 2021 and Physics of Fluids 2021
                      (network predicts the CHANGE of the state; the vector potential is shifted to zero mean)

All the rules come from physical_equations.py.
"""

from dataclasses import dataclass                                   # small containers for the inputs and outputs

import torch                                                        # tensors and automatic gradients
import torch.nn as nn                                               # neural-network layers
import torch.nn.functional as F                                     # padding and pooling functions

from config import Scales                                           # reference scales (for the default time step)
from physics_neural_solver import physical_equations as eq          # all equations live there


# ------------------------------------------------------------------------------------------------------
# What goes in and what comes out
# ------------------------------------------------------------------------------------------------------
@dataclass
class WaterState:                                                   # the water at one moment (what the network learns)
    potential: torch.Tensor                                         # A    (B, 3, X, Y, Z) vector potential, u = curl(A)
    pressure: torch.Tensor                                          # p    (B, 1, X, Y, Z) pressure above atmospheric
    phi: torch.Tensor                                               # phi  (B, 1, X, Y, Z) level set, > 0 in water


@dataclass
class BoxInputs:                                                    # everything else the network needs to know
    solid_mask: torch.Tensor                                        # (1, 1, X, Y, Z) 1 in the wall cells (the border)
    stone_mask: torch.Tensor                                        # (B, 1, X, Y, Z) 1 inside the stone
    stone_velocity_field: torch.Tensor                              # (B, 3, X, Y, Z) stone velocity relative to the box (eq. 13)
    tilt: torch.Tensor                                              # (B,) tilt angle in radians (+ raises the right end)
    tilt_rate: torch.Tensor                                         # (B,) tilt speed  Omega_z
    tilt_acceleration: torch.Tensor                                 # (B,) its rate of change  dOmega_z/dt
    initial_volume: torch.Tensor = None                             # (B,) water volume to keep (eq. 7); None = keep the current one


INPUT_CHANNELS = 15                                                 # phi 1 + A 3 + moving p 1 + walls 1 + stone 1 + stone velocity 3 + gravity 3 + tilt rate 1 + tilt acceleration 1
OUTPUT_CHANNELS = 4                                                 # change of A (3), change of the MOVING pressure (1)
                                                                    # (phi is carried, not predicted)

NETWORK_SIZES = {                                                   # base number of channels (doubles at every level down)
    "small": 16,                                                    # ~1.4 million weights
    "medium": 32,                                                   # ~5.6 million weights
    "large": 64,                                                    # ~22 million weights
}


def still_water_pressure(phi: torch.Tensor) -> torch.Tensor:
    """rho g (depth below the surface), dimensionless (rho = g = 1): phi in the water, 0 in the air (eq. 16)."""
    return phi.clamp(min=0.0)


def moving_pressure(pressure: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """The part of the pressure caused by motion: total minus the still-water part."""
    return pressure - still_water_pressure(phi)


def network_input(state: WaterState, box: BoxInputs) -> torch.Tensor:
    """Stack the water state and the box inputs into one (B, 15, X, Y, Z) tensor."""
    batch = state.phi.shape[0]                                      # number of scenes in the batch
    size = state.phi.shape[2:]                                      # grid size (X, Y, Z)
    gravity = eq.gravity_in_box_frame(box.tilt)                     # (B, 3) gravity seen from inside the box (eq. 3)

    def everywhere(values):                                         # copy one number per scene to every cell
        return values.view(batch, -1, 1, 1, 1).expand(-1, -1, *size)

    return torch.cat([                                              # join along the channel axis
        state.phi,                                                  # 1 channel:  where the water is
        state.potential,                                            # 3 channels: the vector potential (the flow)
        moving_pressure(state.pressure, state.phi),                 # 1 channel:  the moving part of the pressure
        box.solid_mask.expand(batch, -1, -1, -1, -1),               # 1 channel:  where the walls are
        box.stone_mask,                                             # 1 channel:  where the stone is
        box.stone_velocity_field,                                   # 3 channels: how the stone moves
        everywhere(gravity),                                        # 3 channels: gravity direction in the box
        everywhere(box.tilt_rate[:, None]),                         # 1 channel:  tilt speed
        everywhere(box.tilt_acceleration[:, None]),                 # 1 channel:  tilt acceleration
    ], dim=1)


# ------------------------------------------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions, each followed by normalisation and an activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()                                          # set up the PyTorch module
        groups = min(8, out_channels)                               # GroupNorm groups (works with small batches)
        self.layers = nn.Sequential(                                # the layers, applied one after another
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),   # looks at each cell and its 26 neighbours
            nn.GroupNorm(groups, out_channels),                     # keeps the numbers at a stable size
            nn.SiLU(),                                              # smooth activation (allows curved responses)
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),  # second look, now at the new features
            nn.GroupNorm(groups, out_channels),                     # normalise again
            nn.SiLU(),                                              # activate again
        )

    def forward(self, x):
        return self.layers(x)                                       # run the two convolutions


# ------------------------------------------------------------------------------------------------------
# The network
# ------------------------------------------------------------------------------------------------------
class WaterUNet3D(nn.Module):
    """3D U-Net: going down it sees larger and larger regions (big waves), going up it restores the fine
    detail (ripples, splash edges) through the skip connections."""

    def __init__(self, base_channels: int = 32, depth: int = 3):
        super().__init__()                                          # set up the PyTorch module
        self.depth = depth                                          # how many times the grid is halved
        channels = [base_channels * 2 ** level for level in range(depth + 1)]   # e.g. 32, 64, 128, 256

        self.down_blocks = nn.ModuleList()                          # encoder: one ConvBlock per level
        previous = INPUT_CHANNELS                                   # the first block reads the 15 input channels
        for level in range(depth):                                  # for every level going down
            self.down_blocks.append(ConvBlock(previous, channels[level]))   # extract features at this size
            previous = channels[level]                              # the next block reads these features

        self.bottom = ConvBlock(channels[depth - 1], channels[depth])   # the coarsest level: sees the whole box

        self.up_samplers = nn.ModuleList()                          # decoder: doubles the grid size again
        self.up_blocks = nn.ModuleList()                            # decoder: one ConvBlock per level
        for level in reversed(range(depth)):                        # for every level going up
            self.up_samplers.append(nn.ConvTranspose3d(channels[level + 1], channels[level], kernel_size=2, stride=2))  # learned upsampling x2
            self.up_blocks.append(ConvBlock(2 * channels[level], channels[level]))  # merge with the skip connection

        self.output = nn.Conv3d(channels[0], OUTPUT_CHANNELS, kernel_size=1)    # 1x1 convolution to the 4 outputs
        nn.init.zeros_(self.output.weight)                          # start at zero: an untrained network predicts "no change"
        nn.init.zeros_(self.output.bias)                            # (a stable starting point for training)

    def forward(self, x):
        size = x.shape[2:]                                          # remember the grid size (X, Y, Z)
        multiple = 2 ** self.depth                                  # the grid must divide by 2 this many times
        padding = []                                                # F.pad wants (z_low, z_high, y_low, y_high, x_low, x_high)
        for n in reversed(size):                                    # z, then y, then x
            padding += [0, (-n) % multiple]                         # pad at the high end up to the next multiple
        x = F.pad(x, padding, mode="replicate")                     # pad by copying the edge values

        skips = []                                                  # features kept for the way back up
        for block in self.down_blocks:                              # going down
            x = block(x)                                            # features at this level
            skips.append(x)                                         # keep them for the skip connection
            x = F.max_pool3d(x, kernel_size=2)                      # halve the grid in every direction
        x = self.bottom(x)                                          # coarsest level

        for upsample, block, skip in zip(self.up_samplers, self.up_blocks, reversed(skips)):   # going up
            x = upsample(x)                                         # double the grid
            x = block(torch.cat([x, skip], dim=1))                  # merge with the fine detail from the way down

        x = self.output(x)                                          # 4 output channels
        return x[:, :, :size[0], :size[1], :size[2]]                # remove the padding again


# ------------------------------------------------------------------------------------------------------
# The solver step: network + built-in constraints
# ------------------------------------------------------------------------------------------------------
class PhysicsNeuralSolver(nn.Module):
    """One time step of the water:  state at time n  ->  state at time n+1."""

    def __init__(self, network_size: str = "medium", depth: int = 3, time_step: float = None):
        super().__init__()                                          # set up the PyTorch module
        self.network = WaterUNet3D(NETWORK_SIZES[network_size], depth)   # the 3D U-Net
        self.time_step = time_step if time_step is not None else (1.0 / 60.0) / Scales().time   # dt (dimensionless), one frame

    def forward(self, state: WaterState, box: BoxInputs, cell_size: float) -> WaterState:
        change = self.network(network_input(state, box)).float()    # predicted change of A and p (float32 for the physics)

        change_potential = change[:, 0:3]                           # change of the vector potential A
        change_potential = change_potential - change_potential.mean(dim=(2, 3, 4), keepdim=True)  # A is only defined up to a constant: keep its mean fixed
        wall_mask = eq.wall_potential_mask(state.phi.shape[2:], state.phi.device)   # 0 for A's in-wall components
        potential_next = (state.potential + change_potential) * wall_mask   # new A: incompressible (eq. 1), no flow through walls (eq. 10)

        u_mid = 0.5 * (eq.velocity_from_vector_potential(state.potential, cell_size)   # the water velocity during the step
                       + eq.velocity_from_vector_potential(potential_next, cell_size))  # (average of now and next, as in the loss)
        phi_next = eq.carry_surface(state.phi, u_mid, cell_size, self.time_step)   # the surface moves with the water (eq. 5)
        fluid = (1.0 - box.solid_mask) * (1.0 - box.stone_mask)     # cells that can hold water
        target = box.initial_volume                                 # the volume to keep ...
        if target is None:                                          # ... or, if not given, the volume right now
            target = (eq.smoothed_step(state.phi, cell_size) * fluid).sum(dim=(1, 2, 3, 4)) * cell_size ** 3
        phi_next = eq.restore_volume(phi_next, fluid, target, cell_size)   # exactly the same amount of water (eq. 7)

        moving_next = moving_pressure(state.pressure, state.phi) + change[:, 3:4]   # new moving part of the pressure
        pressure_next = moving_next + still_water_pressure(phi_next)  # + the still-water part of the NEW surface
        pressure_next = eq.apply_free_surface_pressure(pressure_next, phi_next, cell_size)  # p = 0 above the water (eq. 8)

        return WaterState(potential=potential_next, pressure=pressure_next, phi=phi_next)   # the water one step later

    def velocity(self, state: WaterState, cell_size: float) -> torch.Tensor:
        return eq.velocity_from_vector_potential(state.potential, cell_size)   # u = curl(A) on the cell faces (eq. 1)


def count_weights(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())               # total number of trainable numbers
