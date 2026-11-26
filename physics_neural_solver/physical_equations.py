"""
physical_equations.py  -  every equation of the water model (PLAN.md), written as code

The network learns three fields on every grid cell: A (velocity u = curl A), p (pressure), phi (water surface).
The equations are not learned; they are enforced:

    Section 1  BUILT-IN CONSTRAINTS   exact by construction              eqs. 1, 8
    Section 2  FORMULAS               used inside the loss               eqs. 3, 4 (+ parts of 2, 11)
    Section 3  LOSS EQUATIONS         residuals pushed towards zero      eqs. 2, 5, 6, 7, 9, 10, 11, 13
    Section 4  STARTING CONDITIONS    calm water                         eqs. 14, 15, 16
    Section 5  STONE (EXACT)          computed exactly, never learned    eqs. 17, 18, 19, 20
    Section 6  ANALYTICAL CHECKS      testing only                       C1 - C5
    Section 0  GRID DERIVATIVES       how derivatives are taken on the grid

Units: dimensionless (length 0.25 m, speed sqrt(g * 0.25 m), pressure rho g * 0.25 m) -> rho = 1, |g| = 1, nu = 1/Re.
Grid: staggered (MAC), attached to the box, one wall cell on every side. Tensors are (batch, channels, X, Y, Z).
      p, phi at cell centres; u_x, u_y, u_z on cell faces; A on cell edges. Positions r are measured from the box centre.
Time: one step n -> n+1 of size dt; the loss uses the average u' = (u_now + u_next) / 2 (IMEX).
"""

import math                                                         # pi, sqrt for the stone and the checks

import torch                                                        # tensors

from config import Scales, WaterConstants                           # physical constants and reference scales

X, Y, Z = 2, 3, 4                                                   # spatial dimensions of a (batch, channels, X, Y, Z) tensor


# ======================================================================================================
# Section 0.  GRID DERIVATIVES
#   The network gives numbers on grid cells, so a derivative d/dx is computed from neighbouring cells.
#   Which neighbours depends on WHERE the result must sit:
#     forward  (f[i+1] - f[i]) / h        -> on the face between i and i+1   (pressure gradient -> velocity faces)
#     backward (f[i] - f[i-1]) / h        -> on the other side               (curl and divergence: exact incompressibility)
#     central  (f[i+1] - f[i-1]) / (2h)   -> at the same cell centre         (gradients at centres)
#     upwind   from the side the flow comes from                            (moving the surface without wiggles)
# ======================================================================================================

def previous_cell(f, dim):
    n = f.size(dim)                                                 # number of cells along this direction
    return torch.cat([f.narrow(dim, 0, 1), f.narrow(dim, 0, n - 1)], dim)   # f[i-1] (the first cell copies itself)


def next_cell(f, dim):
    n = f.size(dim)                                                 # number of cells along this direction
    return torch.cat([f.narrow(dim, 1, n - 1), f.narrow(dim, n - 1, 1)], dim)   # f[i+1] (the last cell copies itself)


def backward_difference(f, dim, h):
    return (f - previous_cell(f, dim)) / h                          # (f[i] - f[i-1]) / h


def forward_difference(f, dim, h):
    return (next_cell(f, dim) - f) / h                              # (f[i+1] - f[i]) / h


def central_difference(f, dim, h):
    return (next_cell(f, dim) - previous_cell(f, dim)) / (2.0 * h)  # (f[i+1] - f[i-1]) / (2h)


def faces_to_centres(velocity):
    ux = 0.5 * (velocity[:, 0:1] + previous_cell(velocity[:, 0:1], X))   # average the two x-faces of each cell
    uy = 0.5 * (velocity[:, 1:2] + previous_cell(velocity[:, 1:2], Y))   # average the two y-faces
    uz = 0.5 * (velocity[:, 2:3] + previous_cell(velocity[:, 2:3], Z))   # average the two z-faces
    return torch.cat([ux, uy, uz], dim=1)                           # velocity at the cell centres


def centres_to_faces(vector):
    vx = 0.5 * (vector[:, 0:1] + next_cell(vector[:, 0:1], X))      # x-component onto the x-faces
    vy = 0.5 * (vector[:, 1:2] + next_cell(vector[:, 1:2], Y))      # y-component onto the y-faces
    vz = 0.5 * (vector[:, 2:3] + next_cell(vector[:, 2:3], Z))      # z-component onto the z-faces
    return torch.cat([vx, vy, vz], dim=1)                           # vector on the faces


def scalar_to_faces(scalar):
    return torch.cat([0.5 * (scalar + next_cell(scalar, d)) for d in (X, Y, Z)], dim=1)   # one scalar onto the x-, y-, z-faces


def faces_open(fluid):
    return torch.cat([torch.minimum(fluid, next_cell(fluid, d)) for d in (X, Y, Z)], dim=1)  # 1 on faces where both sides can hold water


def faces_touching(mask):
    return torch.cat([torch.maximum(mask, next_cell(mask, d)) for d in (X, Y, Z)], dim=1)   # 1 on faces where either side is 1


def gradient_at_centres(scalar, h):
    return torch.cat([central_difference(scalar, d, h) for d in (X, Y, Z)], dim=1)   # (d/dx, d/dy, d/dz) at the centres


def velocity_gradient_at_centres(velocity_c, h):
    return torch.stack([gradient_at_centres(velocity_c[:, i:i + 1], h) for i in range(3)], dim=1)   # G[i, j] = du_i/dx_j


def laplacian(f, h):
    total = -6.0 * f                                                # the cell itself, counted 6 times
    for d in (X, Y, Z):                                             # for each direction
        total = total + next_cell(f, d) + previous_cell(f, d)       # add both neighbours
    return total / (h * h)                                          # sum of (f[i+1] - 2 f[i] + f[i-1]) / h^2


def upwind_advection(scalar, velocity_c, h):
    total = torch.zeros_like(scalar)                                # u . grad(scalar), built up direction by direction
    for axis, d in enumerate((X, Y, Z)):                            # x, y, z
        behind = (3.0 * scalar - 4.0 * previous_cell(scalar, d) + previous_cell(previous_cell(scalar, d), d)) / (2.0 * h)  # 2nd-order slope from behind
        ahead = (-3.0 * scalar + 4.0 * next_cell(scalar, d) - next_cell(next_cell(scalar, d), d)) / (2.0 * h)              # 2nd-order slope from ahead
        u = velocity_c[:, axis:axis + 1]                            # velocity component along this direction
        total = total + torch.where(u > 0, u * behind, u * ahead)   # use the side the flow comes from
    return total


def masked_mean(values, weight):
    weight = weight.expand_as(values)                               # same shape as the values
    return (weight * values).sum() / (weight.sum() + 1e-8)          # average only where the equation applies


# ======================================================================================================
# Section 1.  BUILT-IN CONSTRAINTS  (exact by construction, never in the loss)
# ======================================================================================================

# Eq. 1  MASS CONSERVATION (Navier-Stokes):   div(u) = 0
#   u = velocity. Water can't be squeezed. Built in: u = curl(A), and div(curl A) = 0 is a mathematical identity.
def velocity_from_vector_potential(potential, h):
    ax, ay, az = potential[:, 0:1], potential[:, 1:2], potential[:, 2:3]   # the three components of A (on cell edges)
    ux = backward_difference(az, Y, h) - backward_difference(ay, Z, h)     # u_x = dA_z/dy - dA_y/dz   (on x-faces)
    uy = backward_difference(ax, Z, h) - backward_difference(az, X, h)     # u_y = dA_x/dz - dA_z/dx   (on y-faces)
    uz = backward_difference(ay, X, h) - backward_difference(ax, Y, h)     # u_z = dA_y/dx - dA_x/dy   (on z-faces)
    return torch.cat([ux, uy, uz], dim=1)                                  # velocity on the cell faces


def divergence_of_velocity(velocity, h):                            # used only to CHECK eq. 1 (should be ~0)
    return (backward_difference(velocity[:, 0:1], X, h)             # du_x/dx
            + backward_difference(velocity[:, 1:2], Y, h)           # + du_y/dy
            + backward_difference(velocity[:, 2:3], Z, h))          # + du_z/dz, at the cell centres


# Eq. 8  FREE-SURFACE PRESSURE:   p = 0 where phi <= 0
#   p = pressure above atmospheric, phi = level set. The air above pushes with atmospheric pressure (0 here).
#   Built in: the pressure is set to 0 in every air cell. The exact surface position between cells is handled by
#   the ghost-fluid pressure gradient (Section 2), so water cells keep their full pressure (a smooth step here
#   would wrongly lower the pressure just below the surface and upset the balance with gravity).
def apply_free_surface_pressure(pressure, phi, h):
    return pressure * (phi > 0).float()                            # pressure is switched off above the water


# ======================================================================================================
# Section 2.  FORMULAS  (used inside the loss equations)
# ======================================================================================================

# SMOOTHED STEP:   H(phi) = 1/2 (1 + tanh(phi / 1.5h))
#   1 in water, 0 above it, smooth over ~1.5 cells. Used for the volume (eq. 7) and to mark water cells.
def smoothed_step(phi, h):
    return 0.5 * (1.0 + torch.tanh(phi / (1.5 * h)))                # 0 above the water -> 1 inside it


# Eq. 3  GRAVITY IN THE TILTED BOX:   g_b = R(theta)^T g = |g| (-sin theta, -cos theta, 0)
#   theta = tilt angle about z (+ raises the right end), g = gravity in the room (|g| = 1). Gravity turns as the box tilts.
def gravity_in_box_frame(tilt, gravity_magnitude=1.0):
    return gravity_magnitude * torch.stack([-torch.sin(tilt),       # x: pulls towards the lower end
                                            -torch.cos(tilt),       # y: down
                                            torch.zeros_like(tilt)], dim=1)   # z: no tilt about x


# TILT FORCES (part of eq. 2):   - dOmega/dt x r  -  Omega x (Omega x r)  -  2 Omega x u
#   Omega = (0, 0, w) tilt rate, dOmega/dt its change, r = position from the box centre, u = velocity.
#   Euler, centrifugal and Coriolis forces: exact mechanics of a rotating box.
def tilt_forces(positions, velocity_c, tilt_rate, tilt_acceleration):
    batch = velocity_c.shape[0]                                     # number of scenes
    w = tilt_rate.view(batch, 1, 1, 1, 1)                           # tilt rate, one per scene
    dw = tilt_acceleration.view(batch, 1, 1, 1, 1)                  # tilt acceleration, one per scene
    x, y = positions[:, 0:1], positions[:, 1:2]                     # position from the pivot
    ux, uy = velocity_c[:, 0:1], velocity_c[:, 1:2]                 # velocity at the cell centres
    fx = dw * y + w ** 2 * x + 2.0 * w * uy                         # Euler + centrifugal + Coriolis, x
    fy = -dw * x + w ** 2 * y - 2.0 * w * ux                        # Euler + centrifugal + Coriolis, y
    return torch.cat([fx, fy, torch.zeros_like(fx)], dim=1)         # no force along z


# Eq. 4  SMAGORINSKY TURBULENCE:   nu_t = (C_s h)^2 |S|,   S = 1/2 (grad u + grad u^T),   |S| = sqrt(2 S:S)
#   nu_t = eddy viscosity (friction of swirls smaller than a cell), C_s ~ 0.1, h = cell size, S = strain rate.
def smagorinsky_eddy_viscosity(velocity_gradient, h, smagorinsky_constant):
    strain = 0.5 * (velocity_gradient + velocity_gradient.transpose(1, 2))           # S, symmetric part of grad u
    strain_size = torch.sqrt(2.0 * (strain ** 2).sum(dim=(1, 2)) + 1e-12).unsqueeze(1)   # |S| = sqrt(2 S:S)
    return (smagorinsky_constant * h) ** 2 * strain_size            # nu_t


# PRESSURE GRADIENT WITH THE EXACT SURFACE (part of eq. 2; ghost fluid, Gibou et al. 2002)
#   water-water face: (p1 - p0)/h.  water-air face: p = 0 exactly at phi = 0, a fraction theta = phi_w/(phi_w - phi_a) away.
#   The fix that stopped the classical solver's water from moving on its own.
#   Also returns a scale per face: theta on water-air faces, 1 elsewhere. The momentum equation on a water-air face
#   is multiplied by theta: the same equation with the same solution, but without the factor 1/theta, which would
#   otherwise let a few surface faces dominate the loss (theta can be tiny) and make training very slow.
#   theta is kept >= 0.001 only to avoid dividing by zero (a larger limit, e.g. 0.05, moves the surface and breaks
#   the exact balance of calm water in a tilted box).
SMALLEST_SURFACE_FRACTION = 0.001                                   # lower limit of theta


def pressure_gradient_on_faces(pressure, phi, h):
    gradients, face_scales = [], []                                 # one of each per direction
    for d in (X, Y, Z):                                             # x-, y-, z-faces
        p0, p1 = pressure, next_cell(pressure, d)                   # pressure on both sides of the face
        phi0, phi1 = phi, next_cell(phi, d)                         # level set on both sides
        wet0, wet1 = phi0 > 0, phi1 > 0                             # which side is water
        theta0 = torch.clamp(phi0 / (phi0 - phi1 + 1e-12), SMALLEST_SURFACE_FRACTION, 1.0)   # surface position, water on side 0
        theta1 = torch.clamp(phi1 / (phi1 - phi0 + 1e-12), SMALLEST_SURFACE_FRACTION, 1.0)   # surface position, water on side 1
        gradient = torch.where(wet0 & wet1, (p1 - p0) / h,          # both water: plain difference
                   torch.where(wet0, -p0 / (theta0 * h),            # water -> air: p falls to 0 at the surface
                   torch.where(wet1, p1 / (theta1 * h),             # air -> water: p rises from 0 at the surface
                               torch.zeros_like(p0))))              # both air: no gradient
        scale = torch.where(wet0 & wet1, torch.ones_like(p0),       # water-water face: scale 1
                torch.where(wet0, theta0,                           # water-air face: scale theta
                torch.where(wet1, theta1, torch.ones_like(p0))))
        gradients.append(gradient)                                  # keep this direction
        face_scales.append(scale)
    return torch.cat(gradients, dim=1), torch.cat(face_scales, dim=1)   # (B, 3, ...) on the faces


# WALL LAW (part of eq. 11):   u_t / u_tau = (1/kappa) ln(y u_tau / nu) + B
#   u_t = speed along the wall, u_tau = friction velocity (wall stress tau_w = rho u_tau^2), y = h/2 = distance
#   of the first water cell from the wall, nu = viscosity, kappa = 0.41, B = 5.2. Solved by fixed-point iteration.
def friction_velocity(tangential_speed, distance, viscosity, water: WaterConstants, iterations=6):
    with torch.no_grad():                                           # a coefficient, not something to train through
        u_tau = torch.sqrt(viscosity * tangential_speed / distance + 1e-12)   # first guess (linear law)
        for _ in range(iterations):                                 # refine
            y_plus = torch.clamp(distance * u_tau / viscosity, min=11.0)      # distance in wall units (>= 11: log law region)
            u_tau = tangential_speed / (torch.log(y_plus) / water.von_karman_constant + water.log_law_offset)   # log law
            u_tau = torch.clamp(u_tau, min=1e-8)                    # keep it positive
    return u_tau


# ======================================================================================================
# Section 3.  LOSS EQUATIONS  (each returns residual, weight: weight = 1 where the equation applies)
# ======================================================================================================

# Eq. 2  MOMENTUM (Navier-Stokes, F = m a):
#   du/dt + (u.grad)u = -(1/rho) grad p + div[(nu + nu_t)(grad u + grad u^T)] + g_b + tilt forces
#   u = velocity, p = pressure (new step, exact surface), rho = 1, nu = 1/Re, nu_t = eq. 4, g_b = eq. 3.
def momentum_residual(u_now, u_next, pressure_next, phi_mid, water_faces, positions, gravity,
                      tilt_rate, tilt_acceleration, h, dt, viscosity, water: WaterConstants):
    u_mid = 0.5 * (u_now + u_next)                                  # IMEX average u'
    u_mid_c = faces_to_centres(u_mid)                               # u' at the cell centres
    gradient = velocity_gradient_at_centres(u_mid_c, h)             # G[i, j] = du_i/dx_j
    advection = torch.einsum("bj...,bij...->bi...", u_mid_c, gradient)    # (u.grad)u
    eddy = smagorinsky_eddy_viscosity(gradient, h, water.smagorinsky_constant)   # nu_t (eq. 4)
    viscous = ((viscosity + scalar_to_faces(eddy)) * laplacian(u_mid, h)          # (nu + nu_t) lap(u)
               + centres_to_faces(torch.einsum("bj...,bij...->bi...",              # + grad(nu_t) . (grad u + grad u^T)
                                               gradient_at_centres(eddy, h), gradient + gradient.transpose(1, 2))))
    batch = u_now.shape[0]                                          # number of scenes
    pressure_gradient, face_scale = pressure_gradient_on_faces(pressure_next, phi_mid, h)   # grad p, exact surface
    residual = ((u_next - u_now) / dt                               # du/dt
                + centres_to_faces(advection)                       # + (u.grad)u
                + pressure_gradient                                 # + grad p / rho
                - viscous                                           # - viscosity and turbulence
                - gravity.view(batch, 3, 1, 1, 1)                   # - gravity
                - centres_to_faces(tilt_forces(positions, u_mid_c, tilt_rate, tilt_acceleration)))   # - tilt forces
    return residual * face_scale, water_faces, eddy                 # surface faces scaled by theta; nu_t reused by eq. 11


# Eq. 5  LEVEL-SET TRANSPORT:   d(phi)/dt + u.grad(phi) = 0
#   phi = level set, u = velocity. The surface moves with the water. Checked EVERYWHERE except inside the walls:
#   in the water (otherwise deep water could be turned into "air" unnoticed) and in all of the air (otherwise phi
#   could slowly rise far above the surface, unchecked, and become new "water" out of nothing). In the air, u is the
#   water velocity carried upwards by eq. 9, as classical level-set solvers do.
def level_set_residual(phi_now, phi_next, u_mid_c, solid_mask, h, dt):
    phi_mid = 0.5 * (phi_now + phi_next)                            # IMEX average
    residual = (phi_next - phi_now) / dt + upwind_advection(phi_mid, u_mid_c, h)   # d(phi)/dt + u.grad(phi)
    everywhere = torch.ones_like(phi_now) * (1.0 - solid_mask)      # every cell except the walls
    return residual, everywhere


# Eq. 6  EIKONAL:   |grad(phi)| = 1
#   Keeps phi a true distance so the surface stays sharp (a mathematical tool, not physics). Checked EVERYWHERE
#   except inside the walls, so phi can't form bumps (and new "water") far from the surface.
def eikonal_residual(phi_next, solid_mask, h):
    gradient_size = torch.sqrt((gradient_at_centres(phi_next, h) ** 2).sum(dim=1, keepdim=True) + 1e-12)   # |grad phi|
    everywhere = torch.ones_like(phi_next) * (1.0 - solid_mask)     # every cell except the walls
    return gradient_size - 1.0, everywhere


# Eq. 7  VOLUME CONSERVATION:   sum of H(phi) h^3 = V_0
#   H = smoothed step, h^3 = cell volume, V_0 = starting volume. Water never appears or disappears. One value per scene.
def volume_residual(phi_next, fluid_mask, initial_volume, h):
    volume = (smoothed_step(phi_next, h) * fluid_mask).sum(dim=(1, 2, 3, 4)) * h ** 3   # water volume now (no walls, no stone)
    return (volume - initial_volume) / initial_volume, torch.ones_like(volume)         # relative error


# Eq. 9  FREE-SURFACE NO-SHEAR:   du/dn = 0 above the surface
#   n = surface normal = grad(phi)/|grad(phi)|. Empty air does not drag the water. Checked in ALL of the air (not only
#   just above the surface): this also carries the water velocity upwards into the air along n ("velocity
#   extension", as classical level-set solvers do), so the air can't hold made-up flow that moves phi around.
#   blocked_mask = walls and stone (1 there).
def no_shear_residual(u_next_c, phi_next, blocked_mask, h):
    phi_gradient = gradient_at_centres(phi_next, h)                 # grad phi
    normal = phi_gradient / torch.sqrt((phi_gradient ** 2).sum(dim=1, keepdim=True) + 1e-12)   # unit normal n
    du_dn = torch.einsum("bj...,bij...->bi...", normal, velocity_gradient_at_centres(u_next_c, h))   # (n.grad)u
    all_air = (phi_next < 0).float() * (1.0 - blocked_mask)         # every air cell, no walls, no stone
    return du_dn, all_air


# Eq. 10  WALL NO-PENETRATION:   u.n_w = 0
#   n_w = wall normal. Water can't flow through walls, floor or lid. Checked on the faces between water and wall cells.
def no_penetration_residual(u_next):
    into_walls = []                                                 # the normal velocity on every wall face
    for axis, d in enumerate((X, Y, Z)):                            # walls facing x, y, z
        component = u_next[:, axis:axis + 1]                        # velocity component across these walls
        n = component.size(d)                                       # number of cells along this direction
        into_walls.append(component.narrow(d, 0, 1).flatten(1))     # face between wall cell 0 and the first water cell
        into_walls.append(component.narrow(d, n - 2, 1).flatten(1))  # face between the last water cell and wall cell n-1
    residual = torch.cat(into_walls, dim=1)                         # all wall faces together
    return residual, torch.ones_like(residual)


# Eq. 11  WALL LAW (smooth walls):   u_t/u_tau = (1/kappa) ln(y u_tau/nu) + B,   tau_w = rho u_tau^2
#   Ghost-cell form: in the first water layer, the discrete stress (nu + nu_t)(u_water - u_wall_cell)/h must equal u_tau^2.
def wall_law_residual(u_next_c, eddy_viscosity, h, viscosity, water: WaterConstants):
    residuals = []                                                  # one block per wall
    for normal_axis, d in enumerate((X, Y, Z)):                     # walls facing x, y, z
        n = u_next_c.size(d)                                        # number of cells along this direction
        along_wall = [axis for axis in range(3) if axis != normal_axis]   # the two velocity components along the wall
        for water_index, wall_index in ((1, 0), (n - 2, n - 1)):    # the low wall and the high wall
            u_water = u_next_c.narrow(d, water_index, 1)[:, along_wall]       # velocity along the wall, first water cell
            u_wall_cell = u_next_c.narrow(d, wall_index, 1)[:, along_wall]    # velocity in the wall cell (ghost value)
            nu_total = viscosity + eddy_viscosity.narrow(d, water_index, 1)   # molecular + turbulent viscosity
            speed = u_water.norm(dim=1, keepdim=True)                         # speed along the wall
            u_tau = friction_velocity(speed, 0.5 * h, viscosity, water)       # from the wall law
            discrete_stress = nu_total * (u_water - u_wall_cell) / h          # friction the grid produces
            wall_law_stress = u_tau ** 2 * u_water / (speed + 1e-8)           # friction the wall law wants (along the flow)
            residuals.append((discrete_stress - wall_law_stress).flatten(1))  # the difference
    residual = torch.cat(residuals, dim=1)                          # all walls together
    return residual, torch.ones_like(residual)


# Eq. 13  WATER-STONE BOUNDARY CONDITION:   u = [V_s + omega_s x (x - x_s)] - [Omega x x]   on the stone
#   V_s, omega_s, x_s = stone velocity, spin, centre; Omega x x = the box's own motion. The scene provides this
#   difference as `stone_velocity_field` (at the stone's cell centres). Water can't pass through the stone and moves
#   with it. Checked on every cell FACE that touches a stone cell (as the classical solver does), using the stone's
#   velocity from the stone side of the face. Only faces between the stone and WATER: a stone flying through the air
#   has no water to push (forcing the empty air to move with it would only be a meaningless target).
#   water_cells = 1 in cells holding water (not wall, not stone).
def stone_boundary_residual(u_next, stone_velocity_field, stone_mask, solid_mask, water_cells):
    residuals, weights = [], []                                     # one per velocity component
    for axis, d in enumerate((X, Y, Z)):                            # x-, y-, z-faces
        stone_here, stone_next = stone_mask, next_cell(stone_mask, d)   # is each side of the face stone?
        field = stone_velocity_field[:, axis:axis + 1]              # stone velocity, this component
        target = torch.where(stone_here > 0, field, next_cell(field, d))   # take it from the stone side
        touching = torch.maximum(stone_here, stone_next)            # the face touches the stone
        wet = torch.maximum(water_cells, next_cell(water_cells, d)) # ... and water on the other side
        open_face = (1.0 - solid_mask) * (1.0 - next_cell(solid_mask, d))   # not a wall face
        residuals.append(u_next[:, axis:axis + 1] - target)         # water velocity - stone velocity
        weights.append(touching * wet * open_face)
    return torch.cat(residuals, dim=1), torch.cat(weights, dim=1)


# ALL LOSS EQUATIONS TOGETHER:   L = sum of lambda_i L_i,   L_i = weighted mean of residual_i^2
#   state_now / state_next: objects with .potential, .pressure, .phi (model.WaterState; state_next already has eq. 8)
#   box: object with .solid_mask, .stone_mask, .stone_velocity_field, .tilt, .tilt_rate, .tilt_acceleration (model.BoxInputs)
def loss_terms(state_now, state_next, box, positions, initial_volume, h, dt, scales: Scales = Scales()):
    water = scales.water                                            # physical constants
    viscosity = scales.dimensionless_viscosity                      # nu = 1/Re

    u_now = velocity_from_vector_potential(state_now.potential, h)  # eq. 1: velocity now
    u_next = velocity_from_vector_potential(state_next.potential, h)    # eq. 1: velocity next
    u_mid_c = faces_to_centres(0.5 * (u_now + u_next))              # IMEX average at the centres
    u_next_c = faces_to_centres(u_next)                             # next velocity at the centres
    phi_mid = 0.5 * (state_now.phi + state_next.phi)                # IMEX average of the level set
    fluid = (1.0 - box.solid_mask) * (1.0 - box.stone_mask)         # cells that can hold water
    water_cells = (phi_mid > 0).float() * fluid                     # cells holding water (not wall, not stone)
    water_faces = faces_touching(water_cells) * faces_open(fluid)   # faces next to water, not touching a wall or the stone
    gravity = gravity_in_box_frame(box.tilt)                        # eq. 3

    momentum, momentum_w, eddy = momentum_residual(u_now, u_next, state_next.pressure, phi_mid, water_faces,
                                                   positions, gravity, box.tilt_rate, box.tilt_acceleration,
                                                   h, dt, viscosity, water)                          # eq. 2
    level_set, level_set_w = level_set_residual(state_now.phi, state_next.phi, u_mid_c, box.solid_mask, h, dt)   # eq. 5
    eikonal, eikonal_w = eikonal_residual(state_next.phi, box.solid_mask, h)                           # eq. 6
    volume, volume_w = volume_residual(state_next.phi, fluid, initial_volume, h)                       # eq. 7
    no_shear, no_shear_w = no_shear_residual(u_next_c, state_next.phi, 1.0 - fluid, h)                 # eq. 9 (walls and stone blocked)
    no_penetration, no_penetration_w = no_penetration_residual(u_next)                                 # eq. 10
    wall_law, wall_law_w = wall_law_residual(u_next_c, eddy, h, viscosity, water)                      # eq. 11
    stone, stone_w = stone_boundary_residual(u_next, box.stone_velocity_field, box.stone_mask, box.solid_mask,
                                             water_cells)                                               # eq. 13

    return {                                                        # one number per equation
        "momentum (eq. 2)": masked_mean(momentum ** 2, momentum_w),
        "level set (eq. 5)": masked_mean(level_set ** 2, level_set_w),
        "eikonal (eq. 6)": masked_mean(eikonal ** 2, eikonal_w),
        "volume (eq. 7)": masked_mean(volume ** 2, volume_w),
        "no shear (eq. 9)": masked_mean((no_shear ** 2).sum(dim=1, keepdim=True), no_shear_w),
        "no penetration (eq. 10)": masked_mean(no_penetration ** 2, no_penetration_w),
        "wall law (eq. 11)": masked_mean(wall_law ** 2, wall_law_w),
        "stone boundary (eq. 13)": masked_mean(stone ** 2, stone_w),
    }


# ======================================================================================================
# Section 4.  STARTING CONDITIONS  (calm water at a reset or a water-level change)
#   positions (1, 3, X, Y, Z) from the box centre, depth (B,) = h_0, tilt (B,), box_height = inside height (all dimensionless)
# ======================================================================================================
def starting_state(positions, depth, tilt, box_height):
    batch = depth.shape[0]                                          # number of scenes
    up = -gravity_in_box_frame(tilt).view(batch, 3, 1, 1, 1)        # "up" = opposite to gravity

    # Eq. 14  STARTING SURFACE:   phi_0 = h_0 - y   (y measured along "up")
    #   Flat, calm water at depth h_0, its surface at right angles to gravity.
    surface_point = torch.zeros(batch, 3, 1, 1, 1, device=positions.device)   # a point on the surface ...
    surface_point[:, 1, 0, 0, 0] = -box_height / 2 + depth          # ... at height h_0 above the floor, box middle
    phi_0 = (up * (surface_point - positions)).sum(dim=1, keepdim=True)   # signed distance below the surface

    # Eq. 15  STARTING VELOCITY:   u_0 = 0   ->   A_0 = 0
    #   The water starts at rest (the curl of zero is zero).
    potential_0 = torch.zeros(batch, 3, *positions.shape[2:], device=positions.device)   # no flow

    # Eq. 16  STARTING PRESSURE:   p_0 = rho g (h_0 - y)   (hydrostatic; = phi_0 dimensionless, 0 above the water)
    #   The exact pressure of water at rest: it grows with depth.
    pressure_0 = torch.clamp(phi_0, min=0.0)                        # depth below the surface, 0 above it

    return phi_0, potential_0, pressure_0


# ======================================================================================================
# Section 5.  STONE (EXACT, NOT LEARNED)  -  a solid sphere, dimensionless (water density 1;
#             the stone's density is the ratio density_ratio = rho_s / rho). Works on single values or on
#             batches: vectors are (..., 3) tensors, numbers are (...) tensors.
# ======================================================================================================

ADDED_MASS_COEFFICIENT = 0.5                                        # C_a of a sphere
DRAG_COEFFICIENT = 0.47                                             # C_d of a sphere
RESTITUTION = 0.3                                                   # e: how bouncy the stone is on the walls
WALL_FRICTION = 0.4                                                 # mu_c: Coulomb friction with the walls


# Eq. 19  STONE MASS:   m = rho_s (4/3) pi a^3,   I = (2/5) m a^2
#   rho_s = stone density (300-3000 kg/m^3), a = radius (diameter 5-10 cm), I = moment of inertia (solid sphere).
def stone_volume(radius):
    return (4.0 / 3.0) * math.pi * radius ** 3                      # (4/3) pi a^3


def stone_mass(density_ratio, radius):
    return density_ratio * stone_volume(radius)                     # m (water density = 1)


def stone_inertia(mass, radius):
    return 0.4 * mass * radius ** 2                                 # I


# VOLUME UNDER WATER (used by eq. 17):   V_u = pi c^2 (3a - c) / 3,   c = clamp(a + d, 0, 2a)
#   d = depth of the stone's centre below the local surface, c = height of the sphere cap under water.
def underwater_volume(radius, centre_depth):
    radius = torch.as_tensor(radius, dtype=torch.float32)           # works for numbers and tensors
    centre_depth = torch.as_tensor(centre_depth, dtype=torch.float32)
    cap = torch.minimum(torch.clamp(radius + centre_depth, min=0.0), 2.0 * radius)   # height of the part under water
    return math.pi * cap * cap * (3.0 * radius - cap) / 3.0         # volume of that sphere cap


# Eq. 17  STONE TRANSLATION, WATER FORCE BY MORISON (Morison et al. 1950):
#   m dV_s/dt = m g + F_w + F_c
#   F_w = rho V_u (-g) + rho V_u du/dt + rho C_a V_u (du/dt - dV_s/dt) + 1/2 rho C_d A |u - V_s| (u - V_s)
#         buoyancy       Froude-Krylov    added mass                      drag
#   u = undisturbed water velocity near the stone (from the network), A = pi a^2 x (fraction under water), F_c = contact.
#   The added mass contains dV_s/dt, so it goes to the left:  (m + rho C_a V_u) dV_s/dt = rest of the forces.
#   Returns (force without the drag and without dV_s/dt, drag coefficient k, effective mass), where the drag is
#   written as  -k (V_s - u),  k = 1/2 rho C_d A |u - V_s|,  so that the caller can treat it implicitly (stable).
def morison_force(volume_under_water, radius, gravity, water_velocity, water_acceleration, stone_velocity, mass):
    fraction_under = volume_under_water / stone_volume(radius)      # 0 (in the air) ... 1 (fully under water)
    buoyancy = -volume_under_water.unsqueeze(-1) * gravity          # rho V_u (-g), rho = 1
    froude_krylov_and_added_mass = ((1.0 + ADDED_MASS_COEFFICIENT) * volume_under_water).unsqueeze(-1) * water_acceleration   # rho (1 + C_a) V_u du/dt
    area = math.pi * radius ** 2 * fraction_under                   # frontal area under water
    relative_speed = (water_velocity - stone_velocity).norm(dim=-1)  # |u - V_s|
    drag_coefficient = 0.5 * DRAG_COEFFICIENT * area * relative_speed   # k, so that drag = -k (V_s - u)
    effective_mass = mass + ADDED_MASS_COEFFICIENT * volume_under_water   # m + rho C_a V_u
    return buoyancy + froude_krylov_and_added_mass, drag_coefficient, effective_mass


# Eq. 18  STONE ROTATION:   I d(omega_s)/dt + omega_s x (I omega_s) = T_w + T_c
#   T_w = 0 for a sphere (pressure acts through the centre), T_c = contact torque. For a sphere omega x (I omega) = 0.
def stone_spin_rate(contact_torque, inertia):
    return contact_torque / inertia                                 # d(omega_s)/dt = T_c / I


# Eq. 20  STONE CONTACT:   bounce v_n -> -e v_n,   slide |friction impulse| <= mu_c |normal impulse|
#   v_n = speed into the wall, e = 0.3, mu_c = 0.4. A sphere's contact point reacts with effective mass (2/7) m.
#   Returns the impulses per unit mass (velocity changes): (normal change, largest friction change).
def contact_velocity_changes(normal_speed_into_wall, slip_speed):
    normal_change = (1.0 + RESTITUTION) * normal_speed_into_wall    # stops the stone and bounces it back
    friction_change = torch.minimum(WALL_FRICTION * normal_change, (2.0 / 7.0) * slip_speed)   # Coulomb limit, or stops the slip
    return normal_change, friction_change


# ======================================================================================================
# Section 6.  ANALYTICAL CHECKS  (textbook answers, ONLY for testing the trained solver)
# ======================================================================================================

def check_hydrostatic_pressure(depth_below_surface):               # C1:  p = rho g (h_0 - y),  u = 0
    return depth_below_surface                                      # dimensionless: rho = g = 1


def check_sloshing_period(box_length, depth, mode=1, gravity=1.0):  # C2:  omega_m^2 = g k_m tanh(k_m h_0),  k_m = m pi / L
    k = mode * math.pi / box_length                                 # wave number of the sloshing mode
    return 2.0 * math.pi / math.sqrt(gravity * k * math.tanh(k * depth))   # period = 2 pi / omega


def check_terminal_velocity(density_ratio, radius, gravity=1.0):    # C3:  v_t = sqrt((8/3)(rho_s/rho - 1) g a / C_d)
    return math.sqrt((8.0 / 3.0) * (density_ratio - 1.0) * gravity * radius / DRAG_COEFFICIENT)   # drag balances weight - buoyancy


def check_floating_fraction(density_ratio):                         # C4:  fraction under water = rho_s / rho
    return density_ratio                                            # for a stone lighter than water

# C5:  V(t) = V_0 over long runs - compare eq. 7's volume with the starting volume (no function needed)
