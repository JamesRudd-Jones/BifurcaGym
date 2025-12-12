import jax
import jax.numpy as jnp
import time


jax.config.update("jax_enable_x64", True)


nx = 300
ny = 150


c = jnp.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                            [1, 1], [-1, 1], [-1, -1], [1, -1]])

w = jnp.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=jnp.float64)

centres = jnp.array(((nx // 4, ny // 2),  # Front cylinder
                                  (4 * nx // 10, ny // 2 + 25),  # Top cylinder
                                  (4 * nx // 10, ny // 2 - 25)  # Bottom cylinder
                                 ))
X, Y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny))  # mask for cylinders
opp_idxs = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
radius = 10
reynolds = 100.0
u_inlet = 0.1  # Inlet velocity
cs2 = 1 / 3  # Speed of sound squared
nu = u_inlet * (ny / 6.0) / reynolds  # Kinematic viscosity (derived from Re)
tau = 3 * nu + 0.5  # Relaxation time

mask = jnp.zeros((ny, nx), dtype=bool)
for cx, cy in centres:
    dist = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = mask | (dist <= radius)

mask_solid = mask

# @jax.jit
# def function_1(rho, u):
#     cu = jnp.dot(u, c.T)
#     usqr = jnp.sum(u ** 2, axis=-1, keepdims=True)
#     feq = w * rho * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * usqr)
#     return feq
#
# @jax.jit
# def function_2(rho, u):
#     cu = jnp.einsum('...d,qd->...q', u, c)
#     usqr = jnp.einsum('...d,...d->...', u, u)
#     cu3 = 3.0 * cu
#     usqr = usqr[..., jnp.newaxis]
#     feq = w * rho * (1.0 + cu3 + 0.5 * cu3 ** 2 - 1.5 * usqr)
#     return feq


def get_equilibrium(rho, u):
        # Projects macroscopic velocity onto discrete directions
        cu = jnp.einsum('...d,qd->...q', u, c)
        usqr = jnp.einsum('...d,...d->...', u, u)
        cu3 = 3.0 * cu
        usqr = usqr[..., jnp.newaxis]
        feq = w * rho * (1.0 + cu3 + 0.5 * cu3 ** 2 - 1.5 * usqr)
        return feq


def boundary_conditions(f, action):
    """
    Applies:
    1. Inlet (Left) - Constant Velocity
    2. Outlet (Right) - Outflow
    3. Cylinders - Rotating Bounce-Back (The 'Action')
    """
    # --- 1. Cylinders (Moving Bounce-Back) ---
    # We need the velocity AT the surface of the cylinders
    # Action = [omega1, omega2, omega3] (Angular velocities)

    f_new = f

    # # Iterate over 3 cylinders (Unrolled loop)
    # for i in range(3):
    #     cx, cy = self.centres[i]
    #     omega = action[i]
    #
    #     # Determine velocity of the wall: v = omega x r
    #     # u_wall_x = -omega * (y - cy)
    #     # u_wall_y =  omega * (x - cx)
    #     uw_x = -omega * (self.Y - cy)
    #     uw_y = omega * (self.X - cx)
    #
    #     # We only care about this velocity AT the solid boundary pixels
    #     # Standard bounce-back: f_in(direction) -> f_out(opposite)
    #     # Moving wall correction: f_out = f_in - 2*w*rho*(u_wall . c)/cs2
    #
    #     # Calculate momentum correction term
    #     # Dot product of Wall Velocity and Direction Vectors
    #     # We do this for all 9 directions at once
    #     wall_dot_c = (uw_x[..., None] * self.c[:, 0]) + (uw_y[..., None] * self.c[:, 1])
    #     correction = 2.0 * self.w * 1.0 * (wall_dot_c / self.cs2)  # approx rho=1.0 at wall
    #
    #     # Apply specifically where the mask is solid
    #     # Note: In a full code, we optimize to only do boundary nodes.
    #     # Here we mask the whole grid for simplicity.
    #     f_bounced = f[:, :, self.opp_idxs]  # The standard bounce
    #     f_rotated = f_bounced - correction
    #
    #     # Update ONLY the solid pixels for this cylinder
    #     # Create a specific mask for this cylinder to avoid overlap issues
    #     dist = jnp.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
    #     cyl_mask = (dist <= 10.0)  # Match radius above
    #
    #     f_new = jnp.where(cyl_mask[..., None], f_rotated, f_new)

    # --- 2. Inlet (Left Wall) ---
    # Enforce constant velocity u_inlet
    # (Simplified Zou-He or Equilibrium imposition)

    # Iterate over 3 cylinders (Unrolled loop)
    def cylinder_step(f_new, centres_action_concat):
        # cx, cy = self.centres[i]
        # omega = action[i]
        cx, cy, omega = centres_action_concat

        # Determine velocity of the wall: v = omega x r
        # u_wall_x = -omega * (y - cy)
        # u_wall_y =  omega * (x - cx)
        uw_x = -omega * (Y - cy)
        uw_y = omega * (X - cx)

        # We only care about this velocity AT the solid boundary pixels
        # Standard bounce-back: f_in(direction) -> f_out(opposite)
        # Moving wall correction: f_out = f_in - 2*w*rho*(u_wall . c)/cs2

        # Calculate momentum correction term
        # Dot product of Wall Velocity and Direction Vectors
        # We do this for all 9 directions at once
        wall_dot_c = (uw_x[..., None] * c[:, 0]) + (uw_y[..., None] * c[:, 1])
        correction = 2.0 * w * 1.0 * (wall_dot_c / cs2)  # approx rho=1.0 at wall

        # Apply specifically where the mask is solid
        # Note: In a full code, we optimize to only do boundary nodes.
        # Here we mask the whole grid for simplicity.
        f_bounced = f[:, :, opp_idxs]  # The standard bounce
        f_rotated = f_bounced - correction

        # Update ONLY the solid pixels for this cylinder
        # Create a specific mask for this cylinder to avoid overlap issues
        dist = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        cyl_mask = (dist <= 10.0)  # Match radius above

        f_new = jnp.where(cyl_mask[..., None], f_rotated, f_new)

        return f_new, None

    scan_input = jnp.concatenate((centres, jnp.expand_dims(action, axis=-1)), axis=-1)
    f_new, _ = jax.lax.scan(cylinder_step, f_new, scan_input, 3)

    rho_inlet = 1.0
    u_vec_inlet = jnp.zeros((ny, 1, 2)).at[..., 0].set(u_inlet)
    feq_inlet = get_equilibrium(rho_inlet, u_vec_inlet)
    f_new = f_new.at[:, 0, :].set(feq_inlet[:, 0, :])  # Force the left column to be equilibrium
    f_new = f_new.at[:, -1, :].set(f_new[:, -2, :])  # Zero-gradient (copy second to last column to last column)

    return f_new


@jax.jit
def function_1(f_pre_collision, f_post_collision):
        """
        Calculates Drag and Lift using the Momentum Exchange Method.
        Force = Sum of momentum transferred during bounce-back.
        """
        # Momentum exchange is essentially: (f_in + f_out) * c
        # We sum this over all boundary nodes.

        # This is a simplified proxy:
        # Drag is roughly proportional to the momentum lost in X direction
        # Lift is proportional to momentum lost in Y direction

        # We isolate the solid nodes
        f_solid = jnp.where(mask_solid[..., None], f_post_collision - f_pre_collision, 0.0)

        # Project onto X and Y
        # Sum over all directions and all solid pixels
        fx = jnp.sum(f_solid * c[:, 0])
        fy = jnp.sum(f_solid * c[:, 1])

        return fx, fy


@jax.jit
def function_2(f_pre_collision, f_post_collision):
        """
        Optimized calculation using einsum to avoid intermediate array allocations.
        Calculates Sum(mask * (f_post - f_pre) * c) in a single pass.
        """
        # 1. Compute the momentum change (XLA will fuse this difference into the einsum kernel)
        delta_f = f_post_collision - f_pre_collision

        # 2. Use einsum to contract dimensions efficiently.
        # String '...,...q,qd->d':
        #   '...' : Matches spatial dimensions (Nx, Ny) of the mask and f
        #   'q'   : Matches the discrete velocity direction (Q)
        #   'd'   : Matches the vector components (x, y) of c
        #
        # Operations:
        #   - Broadcasts mask (...) against delta_f (...q)
        #   - Projects delta_f onto c (qd)
        #   - Sums over all spatial (...) and direction (q) indices
        #   - Returns vector d (fx, fy)

        force = jnp.einsum('...,...q,qd->d',
                           mask_solid.astype(delta_f.dtype), # Ensure mask is float
                           delta_f,
                           c)

        return force[0], force[1]


times_1 = []
times_2 = []

# f_init = function_1(jnp.ones((ny, nx, 1)), jnp.zeros((ny, nx, 2)))
# f_init = function_2(jnp.ones((ny, nx, 1)), jnp.zeros((ny, nx, 2)))
f = get_equilibrium(jnp.ones((ny, nx, 1)), jnp.zeros((ny, nx, 2)))

rho_YX1 = jnp.sum(f, axis=-1, keepdims=True)
u = jnp.dot(f, c) / rho_YX1

feq = get_equilibrium(rho_YX1, u)
f_post_coll = f - (f - feq) / tau

action = jnp.zeros(3)
f_boundary = boundary_conditions(f_post_coll, action)

def stream_step(f, idx):
    f = f.at[:, :, idx].set(jnp.roll(f_boundary[:, :, idx], shift=c[idx], axis=(1, 0)))
    return f, None

f_next, _ = jax.lax.scan(stream_step, f_boundary, jnp.arange(1, 9, 1), 8)

drag, lift = function_1(f_post_coll, f_boundary)
drag, lift = function_2(f_post_coll, f_boundary)

for _ in range(3):
    start_time_2 = time.time()
    f_init = function_2(f_post_coll, f_boundary)
    times_2.append(time.time() - start_time_2)

for _ in range(3):
    start_time_1 = time.time()
    f_init = function_1(f_post_coll, f_boundary)
    times_1.append(time.time() - start_time_1)

print(jnp.mean(jnp.array(times_1)))
print(jnp.mean(jnp.array(times_2)))