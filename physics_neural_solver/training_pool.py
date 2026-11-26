"""
training_pool.py  -  where the training states come from (no simulation data)

Following Wandel, Weinmann & Klein (ICLR 2021): a pool of scenes that starts as calm water. Every training step
a random batch is taken out, the network predicts the next step, and the prediction is put BACK into the pool.
So the pool fills with more and more realistic moving water made by the network itself, checked only by the
physics loss. From time to time scenes are reset to calm water, so the network also learns to start from rest.

On top of that, the pool does what a user does in the viewer, depending on the training stage (curriculum):
    stage "calm"   water at rest in a level box                     (learn: stay still, hydrostatic pressure)
    stage "tilt"   + the box tilts to random angles                 (learn: sloshing)
    stage "stone"  + stones are thrown in (slingshot, any density)  (learn: splash, ripples, sinking / floating)

Keeping the pool healthy (lessons from the first coarse run):
    * a scene is reset as soon as its water is no longer good training material: volume off by more than 5 %,
      or water faster than MAX_SPEED (not only when it is NaN)
    * a quarter of the scenes never get a stone, so calm water and sloshing are not forgotten in stage "stone"
    * random resets are rare (a scene lives ~500 steps = ~8 s on average), so long runs are practised
    * the throw speed limit can be lowered (train.py starts the stone stage with slow throws)
"""

from dataclasses import dataclass                                   # simple containers

import torch                                                        # tensors

from config import GridResolution, Scales                           # grids and reference scales
from physics_neural_solver.simulation import Simulation             # scenes that step the tilt and stones exactly


@dataclass
class StageActions:
    """How often the pool does each action, per scene and per training step."""
    tilt_change: float                                              # chance to pick a new tilt angle
    stone_throw: float                                              # chance to throw a new stone
    reset: float                                                    # chance to reset the scene to calm water


STAGES = {
    "calm": StageActions(tilt_change=0.0, stone_throw=0.0, reset=0.002),    # level box, no stones
    "tilt": StageActions(tilt_change=0.01, stone_throw=0.0, reset=0.002),   # random tilting
    "stone": StageActions(tilt_change=0.01, stone_throw=0.01, reset=0.002), # tilting and stones
}
MAX_VOLUME_ERROR = 0.05                                             # reset a scene whose water volume is off by > 5 %
MAX_SPEED_M_S = 6.0                                                 # ... or whose water moves faster than 6 m/s anywhere
STONE_FREE_FRACTION = 0.25                                          # this share of the pool never gets a stone


class TrainingPool:
    def __init__(self, pool_size: int, grid: GridResolution, device, dt: float, scales: Scales = Scales(), seed: int = 0):
        self.generator = torch.Generator(device="cpu").manual_seed(seed)   # reproducible random choices
        self.simulation = Simulation(pool_size, grid, device, dt, scales)  # the scenes
        self.size = pool_size                                       # number of scenes in the pool
        self.device = device                                        # cpu or cuda
        self.limits = self.simulation.limits                        # slider ranges (dimensionless)
        self.stage = "calm"                                         # current curriculum stage
        self.max_speed = MAX_SPEED_M_S / scales.speed               # health limit, dimensionless
        self.stone_free = torch.arange(pool_size, device=device) < int(STONE_FREE_FRACTION * pool_size)   # scenes that never get a stone
        self.max_throw_speed = self.limits.max_throw_speed          # throw speed limit (train.py may lower it)
        self.reset(torch.arange(pool_size, device=device))          # start: all calm water

    # --------------------------------------------------------------------------------------------------
    # Random numbers (on the CPU generator, then moved to the device)
    # --------------------------------------------------------------------------------------------------
    def _uniform(self, low, high, count):
        return (low + (high - low) * torch.rand(count, generator=self.generator)).to(self.device)   # uniform in [low, high)

    def _chance(self, probability):
        return (torch.rand(self.size, generator=self.generator) < probability).to(self.device)     # True with this probability, per scene

    # --------------------------------------------------------------------------------------------------
    # Actions
    # --------------------------------------------------------------------------------------------------
    def reset(self, indices):
        """Calm water at a random depth; the box level in stage "calm", at a random angle otherwise."""
        count = len(indices)                                        # how many scenes to reset
        depth = self._uniform(self.limits.min_depth, self.limits.max_depth, count)   # random water level
        if self.stage == "calm":
            tilt = torch.zeros(count, device=self.device)           # level box
        else:
            tilt = self._uniform(-self.limits.max_tilt, self.limits.max_tilt, count)   # any allowed angle
        self.simulation.reset(indices, depth, tilt)                 # calm water, no stone

    def _new_tilt_targets(self, indices):
        self.simulation.tilt_target[indices] = self._uniform(-self.limits.max_tilt, self.limits.max_tilt, len(indices))   # the box will turn there

    def _throw_stones(self, indices):
        L = self.limits                                             # shorter name
        for scene in indices.tolist():                              # one stone per scene
            radius = float(self._uniform(L.min_stone_radius, L.max_stone_radius, 1))   # random size
            density = float(self._uniform(L.min_density_ratio, L.max_density_ratio, 1))   # random density (floats or sinks)
            x = float(self._uniform(-L.half_size[0] + radius, L.half_size[0] - radius, 1))   # random place along x
            z = float(self._uniform(-L.half_size[2] + radius, L.half_size[2] - radius, 1))   # random place along z
            y = L.half_size[1] - radius - 0.02                      # just under the lid, like the viewer
            speed = float(self._uniform(0.0, self.max_throw_speed, 1))   # slingshot speed (up to the current limit)
            angle = float(self._uniform(0.0, 6.283185, 1))          # slingshot direction (sideways)
            velocity = (speed * torch.cos(torch.tensor(angle)).item(), 0.0, speed * torch.sin(torch.tensor(angle)).item())
            self.simulation.throw_stone(scene, (x, y, z), velocity, radius, density)   # release it

    # --------------------------------------------------------------------------------------------------
    # Used by the training loop
    # --------------------------------------------------------------------------------------------------
    def sample(self, batch_size: int):
        """Pick `batch_size` random scenes: returns (indices, water state, box inputs)."""
        indices = torch.randperm(self.size, generator=self.generator)[:batch_size].to(self.device)   # random scenes, no repeats
        state, box = self.simulation.inputs(indices)                # what the network reads
        return indices, state, box

    def put_back(self, indices, next_water):
        """Store the network's prediction, move tilt and stones exactly, then do random user actions."""
        self.simulation.advance(indices, next_water)                # the pool now holds the predicted water
        actions = STAGES[self.stage]                                # what this stage allows
        chosen = torch.zeros(self.size, dtype=torch.bool, device=self.device)   # only touch the scenes just used
        chosen[indices] = True
        tilt = torch.nonzero(chosen & self._chance(actions.tilt_change)).flatten()   # scenes that get a new tilt target
        if len(tilt):
            self._new_tilt_targets(tilt)
        throw = torch.nonzero(chosen & ~self.stone_free & self._chance(actions.stone_throw)).flatten()   # scenes that get a stone
        if len(throw):
            self._throw_stones(throw)
        unhealthy = torch.zeros(self.size, dtype=torch.bool, device=self.device)   # scenes no longer fit for training
        unhealthy[indices] = self.simulation.unhealthy_scenes(indices, MAX_VOLUME_ERROR, self.max_speed)
        reset = torch.nonzero(chosen & (self._chance(actions.reset) | unhealthy)).flatten()   # random resets + unhealthy scenes
        if len(reset):
            self.reset(reset)
        return len(reset)                                           # how many scenes were reset (for the log)

    def set_stage(self, stage: str):
        self.stage = stage                                          # "calm", "tilt" or "stone"

    def statistics(self):
        """A few numbers to watch in the log."""
        sim = self.simulation                                       # shorter name
        velocity = sim.water.potential                              # (the size of A is a rough measure of motion)
        volume = (sim.initial_volume > 0)                           # scenes with water
        from scene import water_volume                              # local import: only needed here
        stone_mask = torch.zeros_like(sim.water.phi)                # ignore stones for this quick check
        current = water_volume(sim.water.phi, sim.solid, stone_mask, sim.h)   # water volume now
        error = ((current - sim.initial_volume) / sim.initial_volume.clamp(min=1e-6))[volume]   # relative volume change
        age_seconds = sim.steps_since_reset.float() * sim.dt * sim.scales.time   # how long each scene has run
        return {
            "volume_error_mean": float(error.abs().mean()),         # average |volume change|
            "potential_max": float(velocity.abs().max()),           # largest |A| in the pool
            "stones_active": int(sim.stones.active.sum()),          # how many scenes have a stone
            "scene_age_mean_s": float(age_seconds.mean()),          # average age of the scenes (seconds of water time)
            "scene_age_max_s": float(age_seconds.max()),            # oldest scene
        }
