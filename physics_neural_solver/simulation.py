"""
simulation.py  -  a batch of water scenes that can be stepped forward

Each scene has: the water (A, p, phi), the tilt of the box, and (optionally) one stone.
The network moves the WATER. This file moves everything else, exactly:
    * the tilt: the box turns towards its target angle, at most 30 degrees per second (like the viewer slider)
    * the stone: Newton's law with the Morison water force and wall contacts (eqs. 17-20)
Used by the training pool (train.py) and by the evaluation (evaluate.py).
Everything is dimensionless (see config.Scales).
"""

from dataclasses import dataclass                                   # simple containers

import torch                                                        # tensors

from config import GridResolution, Scales                           # grids and reference scales
from scene import calm_water, cell_positions, dimensionless_cell_size, scene_limits, stone_fields, wall_cells, water_volume
from physics_neural_solver import physical_equations as eq          # all equations
from physics_neural_solver.model import BoxInputs, WaterState       # what the network reads and writes

TILT_GAIN_PER_SECOND = 4.0                                          # the box turns towards the target at 4 x (angle difference) per second
STONE_SUBSTEPS = 8                                                  # small stone steps per water step (for the contacts)


@dataclass
class Stones:
    """One stone per scene (it may be switched off)."""
    active: torch.Tensor                                            # (N,) True if the scene has a stone
    centre: torch.Tensor                                            # (N, 3) position in the box frame
    velocity: torch.Tensor                                          # (N, 3) velocity relative to the box
    spin: torch.Tensor                                              # (N, 3) angular velocity
    radius: torch.Tensor                                            # (N,)
    density_ratio: torch.Tensor                                     # (N,) stone density / water density
    surface_height: torch.Tensor                                    # (N,) water surface height near the stone (box y)
    water_velocity: torch.Tensor                                    # (N, 3) undisturbed water velocity near the stone
    water_acceleration: torch.Tensor                                # (N, 3) its rate of change du/dt


class Simulation:
    """N scenes on one grid. `water` holds the network's fields; tilt and stones are stepped exactly."""

    def __init__(self, number_of_scenes: int, grid: GridResolution, device, dt: float, scales: Scales = Scales()):
        self.n = number_of_scenes                                   # how many scenes
        self.device = device                                        # cpu or cuda
        self.dt = dt                                                # one water step (dimensionless)
        self.scales = scales                                        # reference scales
        self.limits = scene_limits(scales)                          # slider ranges (dimensionless)
        self.h = dimensionless_cell_size(grid, scales)              # cell size (dimensionless)
        self.positions = cell_positions(grid, device, scales)       # (1, 3, X, Y, Z) cell centres
        self.solid = wall_cells(grid, device, scales)               # (1, 1, X, Y, Z) wall cells
        size = self.positions.shape[2:]                             # (X, Y, Z)
        zeros = lambda *shape: torch.zeros(*shape, device=device)   # small helper for empty tensors
        self.water = WaterState(potential=zeros(self.n, 3, *size), pressure=zeros(self.n, 1, *size), phi=zeros(self.n, 1, *size))
        self.tilt, self.tilt_target = zeros(self.n), zeros(self.n)  # current and wanted tilt angle
        self.tilt_rate, self.tilt_acceleration = zeros(self.n), zeros(self.n)   # tilt speed and its change
        self.initial_volume = zeros(self.n)                         # water volume at the last reset (eq. 7)
        self.steps_since_reset = torch.zeros(self.n, dtype=torch.long, device=device)   # age of each scene
        self.stones = Stones(active=torch.zeros(self.n, dtype=torch.bool, device=device), centre=zeros(self.n, 3),
                             velocity=zeros(self.n, 3), spin=zeros(self.n, 3), radius=zeros(self.n),
                             density_ratio=zeros(self.n), surface_height=zeros(self.n),
                             water_velocity=zeros(self.n, 3), water_acceleration=zeros(self.n, 3))

    # --------------------------------------------------------------------------------------------------
    # Setting scenes up
    # --------------------------------------------------------------------------------------------------
    def reset(self, indices, depth, tilt):
        """Calm water (eqs. 14-16) at `depth` with the box at `tilt`, no stone, in the scenes `indices`."""
        phi, potential, pressure = calm_water(self.positions, depth, tilt, self.scales)   # flat, at rest, hydrostatic
        self.water.phi[indices] = phi                               # new surface
        self.water.potential[indices] = potential                   # at rest
        self.water.pressure[indices] = pressure                     # hydrostatic pressure
        self.tilt[indices], self.tilt_target[indices] = tilt, tilt  # the box stays at this angle
        self.tilt_rate[indices], self.tilt_acceleration[indices] = 0.0, 0.0   # not turning
        self.stones.active[indices] = False                         # no stone
        no_stone = torch.zeros_like(phi)                            # empty stone mask for the volume
        self.initial_volume[indices] = water_volume(phi, self.solid, no_stone, self.h)   # V_0 for eq. 7
        self.steps_since_reset[indices] = 0                         # a fresh scene

    def throw_stone(self, index, centre, velocity, radius, density_ratio):
        """Put a stone into scene `index` (the viewer's slingshot: released just under the lid)."""
        s = self.stones                                             # shorter name
        s.active[index] = True                                      # the scene now has a stone
        s.centre[index] = torch.as_tensor(centre, device=self.device, dtype=torch.float32)        # where it starts
        s.velocity[index] = torch.as_tensor(velocity, device=self.device, dtype=torch.float32)    # how it is thrown
        s.spin[index] = 0.0                                         # not spinning
        s.radius[index] = radius                                    # its size
        s.density_ratio[index] = density_ratio                      # heavy (sinks) or light (floats)
        s.water_velocity[index] = 0.0                               # forget the previous stone's water data
        s.water_acceleration[index] = 0.0
        s.surface_height[index] = float("-inf")                     # unknown until the next water step

    # --------------------------------------------------------------------------------------------------
    # What the network sees
    # --------------------------------------------------------------------------------------------------
    def inputs(self, indices):
        """Water state and box inputs of the scenes `indices`, ready for the network."""
        s = self.stones                                             # shorter name
        stone_mask, stone_velocity_field = stone_fields(self.positions, s.centre[indices], s.velocity[indices],
                                                        s.spin[indices], s.radius[indices], s.active[indices])   # the stone on the grid
        state = WaterState(potential=self.water.potential[indices], pressure=self.water.pressure[indices],
                           phi=self.water.phi[indices])              # the water now
        box = BoxInputs(solid_mask=self.solid, stone_mask=stone_mask, stone_velocity_field=stone_velocity_field,
                        tilt=self.tilt[indices], tilt_rate=self.tilt_rate[indices],
                        tilt_acceleration=self.tilt_acceleration[indices],
                        initial_volume=self.initial_volume[indices])   # walls, stone, tilt, the volume to keep
        return state, box

    # --------------------------------------------------------------------------------------------------
    # One step forward (after the network has predicted the new water)
    # --------------------------------------------------------------------------------------------------
    def advance(self, indices, next_water: WaterState, maintained: bool = False):
        """Store the network's new water for the scenes `indices`, then move the tilt and the stones exactly.
        maintained=True: phi has already been re-distanced where due (multi-step training does it with gradients)."""
        phi = next_water.phi.detach()                               # the new surface (no gradients in the pool)
        if not maintained:
            phi = self.maintain(indices, phi)                       # keep phi a true distance (eq. 6), every few steps
        self.water.potential[indices] = next_water.potential.detach()   # keep the prediction
        self.water.pressure[indices] = next_water.pressure.detach()
        self.water.phi[indices] = phi
        self._advance_tilt(indices)                                 # the box turns towards its target
        self._advance_stones(indices, next_water)                   # Newton + Morison + contacts
        self.steps_since_reset[indices] += 1                        # one step older

    def maintain(self, indices, phi):
        """phi (B, 1, X, Y, Z) of the scenes `indices` right after a network step. Every REDISTANCE_EVERY steps of a
        scene: make phi a true distance again (eq. 6, exact), then restore the volume exactly (eq. 7) - re-distancing
        may shift the surface by a hair. Works with gradients (used inside multi-step training too)."""
        indices = torch.as_tensor(indices, device=self.device)      # (a list or a tensor)
        due = (self.steps_since_reset[indices] + 1) % eq.REDISTANCE_EVERY == 0   # (B,) scenes whose turn it is
        if not bool(due.any()):
            return phi                                              # nothing to do this step
        s = self.stones                                             # shorter name
        stone_mask, _ = stone_fields(self.positions, s.centre[indices], s.velocity[indices], s.spin[indices],
                                     s.radius[indices], s.active[indices])   # cells inside the stones
        fluid = (1.0 - self.solid) * (1.0 - stone_mask)             # cells that can hold water
        fixed = eq.restore_volume(eq.redistance(phi, self.solid, self.h), fluid, self.initial_volume[indices], self.h)
        return torch.where(due.view(-1, 1, 1, 1, 1), fixed, phi)    # only the scenes whose turn it is

    def _advance_tilt(self, indices):
        gain = TILT_GAIN_PER_SECOND * self.scales.time              # the same 4 per second as the viewer, dimensionless
        difference = self.tilt_target[indices] - self.tilt[indices] # how far from the target angle
        new_rate = torch.clamp(gain * difference, -self.limits.max_tilt_rate, self.limits.max_tilt_rate)   # turn speed, limited
        self.tilt_acceleration[indices] = (new_rate - self.tilt_rate[indices]) / self.dt   # change of the turn speed
        self.tilt_rate[indices] = new_rate                          # new turn speed
        self.tilt[indices] = self.tilt[indices] + new_rate * self.dt   # new angle

    def _advance_stones(self, indices, next_water: WaterState):
        s = self.stones                                             # shorter name
        active = s.active[indices]                                  # which of these scenes have a stone
        if not bool(active.any()):                                  # nothing to do
            return
        index = indices[active] if torch.is_tensor(indices) else torch.as_tensor(indices, device=self.device)[active]   # scenes with a stone
        velocity_c = eq.faces_to_centres(eq.velocity_from_vector_potential(next_water.potential.detach()[active], self.h))   # water velocity at cell centres
        phi = next_water.phi.detach()[active]                       # water surface
        self._read_water_near_stones(index, velocity_c, phi)        # surface height and undisturbed u near each stone
        gravity = eq.gravity_in_box_frame(self.tilt[index])         # (M, 3) gravity in the box frame (eq. 3)
        small_dt = self.dt / STONE_SUBSTEPS                         # stone sub-step
        for _ in range(STONE_SUBSTEPS):                             # small steps for the contacts
            self._stone_substep(index, gravity, small_dt)

    def _read_water_near_stones(self, index, velocity_c, phi):
        """Surface height near each stone (4 columns around it) and the undisturbed water velocity
        (water cells 2.5-3.5 radii away, where the stone's own disturbance has faded)."""
        s, h = self.stones, self.h                                  # shorter names
        half_x, half_y, half_z = self.limits.half_size              # half the inside size of the box
        nx, ny, nz = phi.shape[2:]                                  # grid size
        y_centres = self.positions[0, 1, 0, :, 0]                   # cell-centre heights along y
        for k, scene in enumerate(index.tolist()):                  # one stone at a time (there are few)
            centre, radius = s.centre[scene], s.radius[scene]       # this stone
            offset = float(radius) + 2.0 * h                        # columns just outside the stone
            heights = []                                            # surface height found in each column
            for dx, dz in ((offset, 0.0), (-offset, 0.0), (0.0, offset), (0.0, -offset)):
                i = int(round((float(centre[0]) + dx + half_x) / h + 0.5))   # column index along x
                kz = int(round((float(centre[2]) + dz + half_z) / h + 0.5))  # column index along z
                i, kz = min(max(i, 1), nx - 2), min(max(kz, 1), nz - 2)      # stay inside the box
                column = phi[k, 0, i, :, kz]                        # phi from the floor to the lid
                wet = torch.nonzero(column[1:ny - 1] > 0).flatten() + 1   # water cells in this column
                if len(wet) == 0:                                   # no water in this column
                    continue
                j = int(wet.max())                                  # the highest water cell
                below, above = float(column[j]), float(column[j + 1])   # phi just below and above the surface
                fraction = below / (below - above) if above <= 0 else 0.5   # where phi = 0 between them
                heights.append(float(y_centres[j]) + fraction * h)  # surface height
            new_height = sum(heights) / len(heights) if heights else float("-inf")   # average of the columns
            s.surface_height[scene] = new_height                    # store it

            distance = (self.positions[0] - centre.view(3, 1, 1, 1)).norm(dim=0)   # distance of every cell from the stone
            inner, outer = max(2.5 * float(radius), float(radius) + 2 * h), max(3.5 * float(radius), float(radius) + 4 * h)
            shell = (distance > inner) & (distance < outer) & (phi[k, 0] > 0) & (self.solid[0, 0] == 0)   # water cells in the shell
            if bool(shell.any()):                                   # average the water velocity there
                u = velocity_c[k][:, shell].mean(dim=1)
            else:                                                   # no water nearby (stone in the air)
                u = torch.zeros(3, device=self.device)
            rate = (u - s.water_velocity[scene]) / self.dt          # du/dt from the previous step
            size, limit = rate.norm(), 2.0                          # limit to 2 g (dimensionless g = 1) against noise
            if size > limit:
                rate = rate * (limit / size)
            s.water_acceleration[scene] = 0.7 * s.water_acceleration[scene] + 0.3 * rate   # smoothed du/dt
            s.water_velocity[scene] = u                             # store u

    def _stone_substep(self, index, gravity, small_dt):
        s = self.stones                                             # shorter name
        radius, centre, velocity = s.radius[index], s.centre[index], s.velocity[index]   # the stones being moved
        mass = eq.stone_mass(s.density_ratio[index], radius)        # m (eq. 19)
        inertia = eq.stone_inertia(mass, radius)                    # I (eq. 19)
        tilt_cos = -gravity[:, 1] / gravity.norm(dim=1)             # depth is measured along "up"
        depth = (s.surface_height[index] - centre[:, 1]) * tilt_cos  # centre below the surface (from the current position)
        depth = torch.nan_to_num(depth, neginf=-1e6)                # no surface found: the stone is in the air
        volume_under = eq.underwater_volume(radius, depth)          # V_u, updated every sub-step (no lag)

        force, drag_k, effective_mass = eq.morison_force(volume_under, radius, gravity, s.water_velocity[index],
                                                         s.water_acceleration[index], velocity, mass)   # eq. 17 without drag
        force = force + mass.unsqueeze(1) * gravity                 # + weight
        positions = centre                                          # tilt forces on the stone (same as on the water)
        force = force + mass.unsqueeze(1) * eq.tilt_forces(positions.view(-1, 3, 1, 1, 1), velocity.view(-1, 3, 1, 1, 1),
                                                           self.tilt_rate[index], self.tilt_acceleration[index]).view(-1, 3)
        a = (small_dt * drag_k / effective_mass).unsqueeze(1)       # implicit drag: V_new = (V + dt F/m_eff + a u) / (1 + a)
        free_velocity = velocity + small_dt * force / effective_mass.unsqueeze(1)   # velocity without drag
        velocity = (free_velocity + a * s.water_velocity[index]) / (1.0 + a)        # drag pulls it towards the water velocity
        centre = centre + small_dt * velocity                       # move

        half = torch.tensor(self.limits.half_size, device=self.device)   # half the inside size (x, y, z)
        spin = s.spin[index]                                        # current spin
        for axis in range(3):                                       # walls facing x, y, z
            for side in (-1.0, 1.0):                                # low wall and high wall
                limit = half[axis] - radius                         # how far the centre may go
                hit = side * centre[:, axis] > limit                # touching this wall
                if not bool(hit.any()):
                    continue
                centre[hit, axis] = side * limit[hit]               # push back to the wall
                normal = torch.zeros(3, device=self.device)         # wall normal (pointing out of the box)
                normal[axis] = side
                into_wall = velocity[:, axis] * side                # speed towards the wall
                moving_in = hit & (into_wall > 0)                   # only if it moves into the wall
                if not bool(moving_in.any()):
                    continue
                contact_velocity = velocity + torch.cross(spin, radius.unsqueeze(1) * normal.expand_as(velocity), dim=1)   # velocity of the contact point
                slip = contact_velocity - (contact_velocity @ normal).unsqueeze(1) * normal   # its part along the wall
                slip_speed = slip.norm(dim=1)                       # how fast it slides
                normal_change, friction_change = eq.contact_velocity_changes(into_wall.clamp(min=0), slip_speed)   # eq. 20
                velocity = velocity - (moving_in.float() * normal_change).unsqueeze(1) * normal   # bounce
                direction = slip / slip_speed.clamp(min=1e-9).unsqueeze(1)   # sliding direction
                friction = (moving_in.float() * friction_change).unsqueeze(1) * direction   # friction velocity change
                velocity = velocity - friction                      # friction slows the slide
                spin = spin - torch.cross(radius.unsqueeze(1) * normal.expand_as(friction), friction, dim=1) * (mass / inertia).unsqueeze(1)   # and spins the stone
        s.centre[index], s.velocity[index], s.spin[index] = centre, velocity, spin   # store the new stone state

    # --------------------------------------------------------------------------------------------------
    # Health of the scenes
    # --------------------------------------------------------------------------------------------------
    def broken_scenes(self):
        """Scenes whose water has blown up (NaN or very large values): they should be reset."""
        bad = ~torch.isfinite(self.water.phi).flatten(1).all(dim=1)  # NaN or infinity in phi
        bad |= ~torch.isfinite(self.water.potential).flatten(1).all(dim=1)   # ... or in A
        bad |= self.water.potential.abs().flatten(1).amax(dim=1) > 1e3   # unreasonably large flow
        return bad

    def unhealthy_scenes(self, indices, max_volume_error, max_speed):
        """Which of the scenes `indices` are no longer good training material (True = reset it):
        blown up (NaN), water volume off by more than `max_volume_error` (e.g. 0.05 = 5 %), or water faster than
        `max_speed` anywhere (dimensionless). Without this the pool fills up with broken water that the network
        then keeps training on."""
        bad = self.broken_scenes()[indices]                         # NaN / infinity / huge A
        s = self.stones                                             # shorter name
        stone_mask, _ = stone_fields(self.positions, s.centre[indices], s.velocity[indices], s.spin[indices],
                                     s.radius[indices], s.active[indices])   # cells inside the stones
        phi = torch.nan_to_num(self.water.phi[indices])             # (NaN scenes are already marked bad)
        volume = water_volume(phi, self.solid, stone_mask, self.h)  # water volume now (eq. 7)
        volume_error = (volume - self.initial_volume[indices]).abs() / self.initial_volume[indices].clamp(min=1e-6)
        bad |= volume_error > max_volume_error                      # water appeared or disappeared
        potential = torch.nan_to_num(self.water.potential[indices]) # the flow
        velocity_c = eq.faces_to_centres(eq.velocity_from_vector_potential(potential, self.h))   # u at the centres
        speed = velocity_c.norm(dim=1, keepdim=True) * (1.0 - self.solid)   # |u|, not inside the walls
        bad |= speed.flatten(1).amax(dim=1) > max_speed             # unrealistically fast water
        return bad
