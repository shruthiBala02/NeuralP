"""
The neural water solver: one 3D U-Net.

Input:  the current water state + the stone + the tilt (16 channels, see scene.network_input).
Output: the CHANGE of the vector potential A (3), the pressure p (1) and the level set phi (1).

The last layer starts at zero, so an untrained network predicts "nothing changes".
The grid is padded up to a multiple of 2**depth (the U-Net halves the size `depth` times)
and cropped back afterwards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NETWORK_PRESETS = {  # base channels; the channels double at every level
    "small": 16,
    "medium": 32,
    "large": 64,
}


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            group_norm(out_channels), nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            group_norm(out_channels), nn.SiLU(),
        )

    def forward(self, x):
        return self.layers(x)


class WaterUNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32, depth: int = 3):
        super().__init__()
        self.depth = depth
        channels = [base_channels * 2 ** level for level in range(depth + 1)]

        self.encoder = nn.ModuleList()
        previous = in_channels
        for level in range(depth):
            self.encoder.append(DoubleConv(previous, channels[level]))
            previous = channels[level]
        self.bottleneck = DoubleConv(channels[depth - 1], channels[depth])

        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for level in reversed(range(depth)):
            self.upsample.append(nn.ConvTranspose3d(channels[level + 1], channels[level], kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(2 * channels[level], channels[level]))

        self.output = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        size = x.shape[2:]
        multiple = 2 ** self.depth
        padding = []
        for n in reversed(size):  # F.pad wants (z_lo, z_hi, y_lo, y_hi, x_lo, x_hi)
            padding += [0, (-n) % multiple]
        x = F.pad(x, padding, mode="replicate")

        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.bottleneck(x)
        for upsample, block, skip in zip(self.upsample, self.decoder, reversed(skips)):
            x = block(torch.cat([upsample(x), skip], dim=1))
        x = self.output(x)
        return x[:, :, :size[0], :size[1], :size[2]]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
