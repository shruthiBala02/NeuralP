"""
Finite differences on a staggered (MAC) grid.

Tensors have shape (batch, channels, X, Y, Z). Every array has the same size;
the position a value represents depends on the quantity:

    p, phi              cell centres         (i,     j,     k)
    u_x / u_y / u_z     cell faces           (i+1/2, j, k) / (i, j+1/2, k) / (i, j, k+1/2)
    A_x / A_y / A_z     cell edges           (i, j+1/2, k+1/2) / (i+1/2, j, k+1/2) / (i+1/2, j+1/2, k)

With this layout u = curl(A) uses only backward differences, and its divergence
at the cell centres is exactly zero (up to rounding).
At the outer edge of the array, values are copied from the nearest cell.
"""

import torch

X, Y, Z = 2, 3, 4  # spatial dimensions of a (batch, channels, X, Y, Z) tensor


def previous(f: torch.Tensor, dim: int) -> torch.Tensor:
    """g[i] = f[i - 1]   (the first entry is copied)."""
    n = f.size(dim)
    return torch.cat([f.narrow(dim, 0, 1), f.narrow(dim, 0, n - 1)], dim)


def following(f: torch.Tensor, dim: int) -> torch.Tensor:
    """g[i] = f[i + 1]   (the last entry is copied)."""
    n = f.size(dim)
    return torch.cat([f.narrow(dim, 1, n - 1), f.narrow(dim, n - 1, 1)], dim)


def backward_difference(f, dim, h):
    return (f - previous(f, dim)) / h


def forward_difference(f, dim, h):
    return (following(f, dim) - f) / h


def central_difference(f, dim, h):
    return (following(f, dim) - previous(f, dim)) / (2.0 * h)


def velocity_from_potential(potential: torch.Tensor, h: float) -> torch.Tensor:
    """u = curl(A) on cell faces. potential: (B, 3, X, Y, Z) on cell edges -> (B, 3, X, Y, Z) on faces."""
    ax, ay, az = potential[:, 0:1], potential[:, 1:2], potential[:, 2:3]
    ux = backward_difference(az, Y, h) - backward_difference(ay, Z, h)
    uy = backward_difference(ax, Z, h) - backward_difference(az, X, h)
    uz = backward_difference(ay, X, h) - backward_difference(ax, Y, h)
    return torch.cat([ux, uy, uz], dim=1)


def divergence_of_face_velocity(velocity: torch.Tensor, h: float) -> torch.Tensor:
    """div(u) at cell centres (used only to check that u = curl(A) is divergence-free)."""
    return (backward_difference(velocity[:, 0:1], X, h)
            + backward_difference(velocity[:, 1:2], Y, h)
            + backward_difference(velocity[:, 2:3], Z, h))


def faces_to_centres(velocity: torch.Tensor) -> torch.Tensor:
    """Average each face velocity component to the cell centres."""
    ux = 0.5 * (velocity[:, 0:1] + previous(velocity[:, 0:1], X))
    uy = 0.5 * (velocity[:, 1:2] + previous(velocity[:, 1:2], Y))
    uz = 0.5 * (velocity[:, 2:3] + previous(velocity[:, 2:3], Z))
    return torch.cat([ux, uy, uz], dim=1)


def centres_to_faces(vector_at_centres: torch.Tensor) -> torch.Tensor:
    """Average a 3-component vector from cell centres to the matching faces."""
    vx = 0.5 * (vector_at_centres[:, 0:1] + following(vector_at_centres[:, 0:1], X))
    vy = 0.5 * (vector_at_centres[:, 1:2] + following(vector_at_centres[:, 1:2], Y))
    vz = 0.5 * (vector_at_centres[:, 2:3] + following(vector_at_centres[:, 2:3], Z))
    return torch.cat([vx, vy, vz], dim=1)


def scalar_centres_to_faces(scalar: torch.Tensor) -> torch.Tensor:
    """Average a scalar from cell centres to the x, y and z faces -> (B, 3, X, Y, Z)."""
    return torch.cat([0.5 * (scalar + following(scalar, X)),
                      0.5 * (scalar + following(scalar, Y)),
                      0.5 * (scalar + following(scalar, Z))], dim=1)


def pressure_gradient_on_faces(pressure: torch.Tensor, h: float) -> torch.Tensor:
    """grad(p) on the faces where each velocity component lives."""
    return torch.cat([forward_difference(pressure, X, h),
                      forward_difference(pressure, Y, h),
                      forward_difference(pressure, Z, h)], dim=1)


def central_gradient(scalar: torch.Tensor, h: float) -> torch.Tensor:
    """grad of a cell-centred scalar at the cell centres -> (B, 3, X, Y, Z)."""
    return torch.cat([central_difference(scalar, X, h),
                      central_difference(scalar, Y, h),
                      central_difference(scalar, Z, h)], dim=1)


def velocity_gradient(velocity_at_centres: torch.Tensor, h: float) -> torch.Tensor:
    """Gradient tensor G[i, j] = d u_i / d x_j at cell centres -> (B, 3, 3, X, Y, Z)."""
    rows = [central_gradient(velocity_at_centres[:, i:i + 1], h) for i in range(3)]
    return torch.stack(rows, dim=1)


def laplacian(f: torch.Tensor, h: float) -> torch.Tensor:
    """7-point Laplacian of every channel."""
    total = -6.0 * f
    for dim in (X, Y, Z):
        total = total + following(f, dim) + previous(f, dim)
    return total / (h * h)


def second_order_upwind_advection(scalar, velocity_at_centres, h):
    """u . grad(scalar) with second-order upwind differences (used for the level set, eq. 5)."""
    total = torch.zeros_like(scalar)
    for axis, dim in enumerate((X, Y, Z)):
        f_m1, f_m2 = previous(scalar, dim), previous(previous(scalar, dim), dim)
        f_p1, f_p2 = following(scalar, dim), following(following(scalar, dim), dim)
        backward = (3.0 * scalar - 4.0 * f_m1 + f_m2) / (2.0 * h)
        forward = (-3.0 * scalar + 4.0 * f_p1 - f_p2) / (2.0 * h)
        u = velocity_at_centres[:, axis:axis + 1]
        total = total + torch.where(u > 0, u * backward, u * forward)
    return total
