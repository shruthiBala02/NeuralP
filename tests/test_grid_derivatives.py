"""
Tests of Section 0 (grid derivatives): every operator is checked on functions whose derivative is known exactly.
Only cells at least 2 away from the array edge are compared (the edge copies its neighbour on purpose).
"""

import torch                                                        # tensors

from physics_neural_solver import physical_equations as eq          # the operators under test

INNER = (slice(None), slice(None), slice(2, -2), slice(2, -2), slice(2, -2))   # cells away from the edges


def test_differences_of_a_linear_function_are_exact(geometry):
    positions, _, h = geometry                                      # cell centres and cell size
    f = 3.0 * positions[:, 0:1] - 2.0 * positions[:, 1:2] + 0.5 * positions[:, 2:3]   # f = 3x - 2y + 0.5z
    for dim, slope in ((eq.X, 3.0), (eq.Y, -2.0), (eq.Z, 0.5)):     # each direction
        for difference in (eq.forward_difference, eq.backward_difference, eq.central_difference):
            result = difference(f, dim, h)[INNER]                   # derivative from neighbouring cells
            assert torch.allclose(result, torch.full_like(result, slope), atol=1e-4)   # exact for a straight line


def test_central_difference_of_a_quadratic_is_exact(geometry):
    positions, _, h = geometry
    x = positions[:, 0:1]                                           # x coordinate of every cell
    result = eq.central_difference(x ** 2, eq.X, h)[INNER]          # d(x^2)/dx
    assert torch.allclose(result, (2.0 * x)[INNER], atol=1e-4)      # = 2x exactly (central differences are 2nd order)


def test_laplacian_of_r_squared_is_six(geometry):
    positions, _, h = geometry
    r2 = (positions ** 2).sum(dim=1, keepdim=True)                  # x^2 + y^2 + z^2
    result = eq.laplacian(r2, h)[INNER]                             # its Laplacian
    assert torch.allclose(result, torch.full_like(result, 6.0), atol=1e-2)   # = 2 + 2 + 2


def test_curl_of_a_known_potential(geometry):
    positions, _, h = geometry
    edge_x = positions[:, 0:1] + 0.5 * h                            # A_z lives on edges (i+1/2, j+1/2, k)
    edge_y = positions[:, 1:2] + 0.5 * h
    potential = torch.zeros(1, 3, *positions.shape[2:])
    potential[:, 2:3] = edge_x * edge_y                             # A = (0, 0, x y)
    velocity = eq.velocity_from_vector_potential(potential, h)      # u = curl(A) = (dA_z/dy, -dA_z/dx, 0) = (x, -y, 0)
    face_x_of_ux = positions[:, 0:1] + 0.5 * h                      # u_x lives on the x-face (i+1/2, j, k)
    face_y_of_uy = positions[:, 1:2] + 0.5 * h                      # u_y lives on the y-face (i, j+1/2, k)
    assert torch.allclose(velocity[:, 0:1][INNER], face_x_of_ux[INNER], atol=1e-4)    # u_x = x
    assert torch.allclose(velocity[:, 1:2][INNER], -face_y_of_uy[INNER], atol=1e-4)   # u_y = -y
    assert torch.allclose(velocity[:, 2:3][INNER], torch.zeros_like(velocity[:, 2:3][INNER]), atol=1e-5)   # u_z = 0


def test_curl_is_always_divergence_free(geometry):
    positions, _, h = geometry
    potential = torch.randn(2, 3, *positions.shape[2:])             # any random potential
    velocity = eq.velocity_from_vector_potential(potential, h)      # u = curl(A)
    divergence = eq.divergence_of_velocity(velocity, h)[INNER]      # div(u)
    scale = velocity.abs().max() / h                                # size of the individual derivative terms
    assert divergence.abs().max() < 1e-5 * scale                    # zero up to rounding: eq. 1 is built in


def test_face_and_centre_averaging(geometry):
    positions, _, _ = geometry
    constant = torch.ones(1, 3, *positions.shape[2:]) * torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1, 1)
    assert torch.allclose(eq.faces_to_centres(constant), constant)  # averaging a constant changes nothing
    assert torch.allclose(eq.centres_to_faces(constant), constant)


def test_upwind_advection_of_a_linear_field_is_exact(geometry):
    positions, _, h = geometry
    phi = 1.0 - positions[:, 1:2] + 0.3 * positions[:, 0:1]         # phi = 1 - y + 0.3 x
    for velocity_value in (+0.7, -0.7):                             # flow from both sides
        velocity = torch.zeros(1, 3, *positions.shape[2:])
        velocity[:, 0] = velocity_value                             # u_x
        velocity[:, 1] = -0.4 * velocity_value                      # u_y
        result = eq.upwind_advection(phi, velocity, h)[INNER]       # u . grad(phi)
        expected = velocity_value * 0.3 + (-0.4 * velocity_value) * (-1.0)   # u_x * 0.3 + u_y * (-1)
        assert torch.allclose(result, torch.full_like(result, expected), atol=1e-4)


def test_masked_mean_ignores_cells_with_zero_weight():
    values = torch.tensor([1.0, 100.0, 3.0])                        # the middle value should be ignored
    weight = torch.tensor([1.0, 0.0, 1.0])
    assert abs(float(eq.masked_mean(values, weight)) - 2.0) < 1e-6  # (1 + 3) / 2
