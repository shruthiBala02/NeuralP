"""
symmetry.py  -  mirror a scene front-to-back (z -> -z)

The box is symmetric front-to-back, and the tilt turns about the z-axis, so a mirrored scene must move exactly like
the mirror image of the original. Training on mirrored copies doubles the variety of the training states for free
and pushes the network towards physically symmetric behaviour. (Physics only: no data is added.)

Where each quantity lives on the grid decides how it is mirrored (z index k, nz cells including the two walls):
    at the cell centre in z     phi, p, masks, A_z, u_x, u_y, stone velocity     k -> nz-1-k
    half a cell further in z    A_x, A_y (cell edges), u_z (z-faces)             k -> nz-2-k  (the last plane stays)
and the sign:
    u_z and the z-part of the stone velocity flip sign (they are ordinary vectors)
    A_x and A_y flip sign, A_z does not (A is the potential whose curl is u: an "axial" vector)
    stone spin: x and y parts flip sign, z part does not (also axial)
The tilt (about z), gravity and the starting volume do not change.
"""

import torch                                                        # tensors

from physics_neural_solver.model import BoxInputs, WaterState       # what is mirrored

Z = 4                                                               # the z axis of a (batch, channels, X, Y, Z) tensor


def flip_centred(field):
    """A field at the cell centres (in z): k -> nz-1-k."""
    return torch.flip(field, dims=[Z])


def flip_staggered(field):
    """A field half a cell further along z: k -> nz-2-k; the last plane (outside the box) stays where it is."""
    inside = torch.flip(field[..., :-1], dims=[Z])                  # planes 0 .. nz-2, reversed
    return torch.cat([inside, field[..., -1:]], dim=Z)              # + the unused last plane


def mirror_potential(potential):
    """A = (A_x, A_y, A_z):  A_x, A_y staggered in z and sign-flipped; A_z centred, sign kept."""
    ax = -flip_staggered(potential[:, 0:1])                         # edge (x_i, y_j+h/2, z_k+h/2)
    ay = -flip_staggered(potential[:, 1:2])                         # edge (x_i+h/2, y_j, z_k+h/2)
    az = flip_centred(potential[:, 2:3])                            # edge (x_i+h/2, y_j+h/2, z_k)
    return torch.cat([ax, ay, az], dim=1)


def mirror_velocity_faces(velocity):
    """u on the faces: u_x, u_y centred in z (sign kept); u_z on z-faces, staggered and sign-flipped."""
    ux = flip_centred(velocity[:, 0:1])                             # x-face (x_i+h/2, y_j, z_k)
    uy = flip_centred(velocity[:, 1:2])                             # y-face (x_i, y_j+h/2, z_k)
    uz = -flip_staggered(velocity[:, 2:3])                          # z-face (x_i, y_j, z_k+h/2)
    return torch.cat([ux, uy, uz], dim=1)


def mirror_state(state: WaterState) -> WaterState:
    return WaterState(potential=mirror_potential(state.potential),  # the flow
                      pressure=flip_centred(state.pressure),        # the pressure
                      phi=flip_centred(state.phi))                  # the surface


def mirror_box(box: BoxInputs) -> BoxInputs:
    field = flip_centred(box.stone_velocity_field)                  # stone velocity at the cell centres
    field = torch.cat([field[:, 0:2], -field[:, 2:3]], dim=1)       # its z-part flips sign
    return BoxInputs(solid_mask=flip_centred(box.solid_mask),       # the walls (symmetric anyway)
                     stone_mask=flip_centred(box.stone_mask),       # where the stone is
                     stone_velocity_field=field,                    # how it moves
                     tilt=box.tilt, tilt_rate=box.tilt_rate,        # tilt about z: unchanged
                     tilt_acceleration=box.tilt_acceleration,
                     initial_volume=box.initial_volume)             # the same amount of water


def mirror_some(state: WaterState, box: BoxInputs, chosen: torch.Tensor):
    """Mirror only the scenes where `chosen` (B,) is True; the others stay as they are."""
    if not bool(chosen.any()):
        return state, box                                           # nothing to do
    m_state, m_box = mirror_state(state), mirror_box(box)           # everything mirrored ...
    pick = lambda a, b: torch.where(chosen.view(-1, 1, 1, 1, 1), b, a)   # ... then keep the chosen scenes
    state = WaterState(pick(state.potential, m_state.potential), pick(state.pressure, m_state.pressure),
                       pick(state.phi, m_state.phi))
    box = BoxInputs(solid_mask=box.solid_mask, stone_mask=pick(box.stone_mask, m_box.stone_mask),
                    stone_velocity_field=pick(box.stone_velocity_field, m_box.stone_velocity_field),
                    tilt=box.tilt, tilt_rate=box.tilt_rate, tilt_acceleration=box.tilt_acceleration,
                    initial_volume=box.initial_volume)
    return state, box
