import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial


width = 15.0
length = 15.0
fluid_grid_size = 128

x_coords, y_coords = jnp.meshgrid(jnp.arange(fluid_grid_size),
                                  jnp.arange(fluid_grid_size))


@jax.jit
def vortex_force(x_world, y_world, t, amplitude=5.0):
    center_x = fluid_grid_size / 2
    center_y = fluid_grid_size / 2

    dx_world = x_world - center_x
    dy_world = y_world - center_y

    force_u = -dy_world * amplitude * 1e-2
    force_v = dx_world * amplitude * 1e-2

    return force_u, force_v


@jax.jit
def vortex_force_new(x_world, y_world, t, amplitude=5.0):
    center_x = fluid_grid_size / 2
    center_y = fluid_grid_size / 2

    dx_world = x_world - center_x
    dy_world = y_world - center_y

    force_u = -dy_world * amplitude * 1e-2
    force_v = dx_world * amplitude * 1e-2

    return force_u, force_v


@partial(jax.jit, static_argnums=(3,))
def sinusoidal_wave_force(x_world, y_world, t, num_waves=5, key=jrandom.key(0)):
    force_u = jnp.zeros_like(x_world, dtype=jnp.float32)
    force_v = jnp.zeros_like(y_world, dtype=jnp.float32)

    # Generate fixed random parameters for this test
    k1, k2, k3, k4, k5, k6 = jrandom.split(key, 6)

    amps = jrandom.uniform(k1, (num_waves,)) * 2.0
    kxs = jrandom.uniform(k2, (num_waves,), minval=0.2, maxval=0.7)
    kys = jrandom.uniform(k3, (num_waves,), minval=0.2, maxval=0.7)
    omegas = jrandom.uniform(k4, (num_waves,), minval=0.1, maxval=0.5)
    phases = jrandom.uniform(k5, (num_waves,)) * 2 * jnp.pi
    dir_angles = jrandom.uniform(k6, (num_waves,)) * 2 * jnp.pi

    wave_params = jnp.stack([amps, kxs, kys, omegas, phases, dir_angles], axis=1)

    def apply_wave(carry, params):
        force_u_carry, force_v_carry = carry
        amp, kx, ky, omega, phase, dir_angle = params

        wave_pattern = amp * jnp.sin(kx * x_world + ky * y_world + omega * t + phase)

        force_u_carry += wave_pattern * jnp.cos(dir_angle)
        force_v_carry += wave_pattern * jnp.sin(dir_angle)
        return (force_u_carry, force_v_carry), None

    (force_u, force_v), _ = jax.lax.scan(apply_wave, (force_u, force_v), wave_params)

    return force_u, force_v


@jax.jit
def horizontal_flow_force(x_world, y_world, t, amplitude=5.0, speed=2.0):
    """
    A simple horizontal flow from left to right, which oscillates in strength over time.
    """
    force_u = jnp.full_like(x_world, amplitude * (1 + jnp.sin(t * speed)))
    force_v = jnp.zeros_like(y_world)
    return force_u, force_v


@jax.jit
def dual_vortex_force(x_world, y_world, t, amplitude=8.0):
    # TODO this should cancel out
    center_x1 = fluid_grid_size * 0.25
    center_y1 = fluid_grid_size / 2
    dx1 = x_world - center_x1
    dy1 = y_world - center_y1
    force_u1 = -dy1 * amplitude * 1e-2
    force_v1 = dx1 * amplitude * 1e-2

    center_x2 = fluid_grid_size * 0.75
    center_y2 = fluid_grid_size / 2
    dx2 = x_world - center_x2
    dy2 = y_world - center_y2
    force_u2 = dy2 * amplitude * 1e-2  # Flipped sign for opposite spin
    force_v2 = -dx2 * amplitude * 1e-2  # Flipped sign

    return force_u1 + force_u2, force_v1 + force_v2


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def moving_eddies_force(x_world, y_world, t, num_eddies=10, amplitude=10.0, radius=20, max_speed=2,
                        key=jrandom.PRNGKey(42)):
    force_u = jnp.zeros_like(x_world, dtype=jnp.float32)
    force_v = jnp.zeros_like(y_world, dtype=jnp.float32)

    # Generate fixed random parameters for eddies
    key, _key = jrandom.split(key)
    initial_pos_x = jrandom.uniform(_key, (num_eddies,)) * fluid_grid_size
    key, _key = jrandom.split(key)
    initial_pos_y = jrandom.uniform(_key, (num_eddies,)) * fluid_grid_size

    key, _key = jrandom.split(key)
    velocities_x = jrandom.uniform(_key, (num_eddies,), minval=-max_speed, maxval=max_speed)
    key, _key = jrandom.split(key)
    velocities_y = jrandom.uniform(_key, (num_eddies,), minval=-max_speed, maxval=max_speed)

    # Random spin direction for each eddy
    key, _key = jrandom.split(key)
    spin = jrandom.choice(_key, jnp.array([-1.0, 1.0]), shape=(num_eddies,))

    eddy_params = jnp.stack([initial_pos_x, initial_pos_y, velocities_x, velocities_y, spin], axis=1)

    def apply_eddy(carry, params):
        force_u_carry, force_v_carry = carry
        ix, iy, vx, vy, s = params

        # Update eddy center position based on time
        center_x = ix + vx * t
        center_y = iy + vy * t

        # Wrap eddies around the domain for continuous flow
        center_x = jnp.mod(center_x, fluid_grid_size)
        center_y = jnp.mod(center_y, fluid_grid_size)

        dx = x_world - center_x
        dy = y_world - center_y

        # Vortex force
        vortex_u = -dy * 1e-2
        vortex_v = dx * 1e-2

        # Gaussian falloff for localized effect
        distance_sq = dx ** 2 + dy ** 2
        falloff = jnp.exp(-distance_sq / (radius ** 2))

        force_u_carry += vortex_u * falloff * amplitude * s
        force_v_carry += vortex_v * falloff * amplitude * s
        return (force_u_carry, force_v_carry), None

    (force_u, force_v), _ = jax.lax.scan(apply_eddy, (force_u, force_v), eddy_params)

    return force_u, force_v


def visualise_force_field(force_func, t=0.0, **kwargs):
    batch_t = jnp.arange(10)
    force_u, force_v = jax.vmap(force_func, in_axes=(None, None, 0))(x_coords, y_coords, batch_t)
    # force_u, force_v = force_func(x_coords, y_coords, t, **kwargs)

    x_fine = jnp.linspace(0, width, fluid_grid_size)
    y_fine = jnp.linspace(0, length, fluid_grid_size)
    X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine)

    coarse_res = 15  # 25
    x_coarse = jnp.linspace(0, width, coarse_res)
    y_coarse = jnp.linspace(0, length, coarse_res)
    X_coarse, Y_coarse = jnp.meshgrid(x_coarse, y_coarse)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, width)
    ax.set_ylim(0, length)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_aspect('equal', adjustable='box')

    def _get_current_at_point(x, y, u_grid, v_grid):
        grid_x = x / width * fluid_grid_size
        grid_y = y / length * fluid_grid_size

        # Coordinates for interpolation
        coords = jnp.array([[grid_y], [grid_x]])

        # Bilinearly interpolate the velocity from the fluid grid
        u_interp = jsp.ndimage.map_coordinates(u_grid, coords, order=1, mode='wrap')[0]
        v_interp = jsp.ndimage.map_coordinates(v_grid, coords, order=1, mode='wrap')[0]

        return jnp.array([u_interp, v_interp])

    vmap_current_spatial = jax.vmap(jax.vmap(_get_current_at_point, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None))

    initial_current_fine = vmap_current_spatial(x_fine, y_fine, force_u[0], force_v[0])
    initial_mag_fine = jnp.linalg.norm(initial_current_fine, axis=-1)
    initial_mag_fine = jnp.round(initial_mag_fine, decimals=3)  # stops tiny errors when flow map is constant

    initial_current_coarse = vmap_current_spatial(x_coarse, y_coarse, force_u[0], force_v[0])
    initial_U, initial_V = initial_current_coarse[:, :, 0], initial_current_coarse[:, :, 1]

    pcm = ax.pcolormesh(X_fine, Y_fine, initial_mag_fine, cmap='plasma', shading='auto', zorder=1, alpha=0.7)
    arrow = ax.quiver(X_coarse, Y_coarse, initial_U, initial_V, color='white', angles='xy', scale_units='xy',
                      scale=1.5, width=0.004, zorder=2)
    ax.legend(loc='upper left')
    fig.colorbar(pcm, ax=ax, shrink=0.8, label='Current Magnitude')
    plt.show()

    def update(frame):
        current_vectors_fine = vmap_current_spatial(x_fine, y_fine, force_u[frame], force_v[frame])
        magnitude_fine = jnp.linalg.norm(current_vectors_fine, axis=-1)

        current_vectors_coarse = vmap_current_spatial(x_coarse, y_coarse, force_u[frame], force_v[frame])
        U_coarse = current_vectors_coarse[:, :, 0]
        V_coarse = current_vectors_coarse[:, :, 1]

        pcm.set_array(magnitude_fine.ravel())
        arrow.set_UVC(U_coarse, V_coarse)

        return pcm, arrow

    # Create the animation
    anim = animation.FuncAnimation(fig,
                                   update,
                                   frames=100,
                                   interval=500,
                                   blit=True)
    anim.save("../../../animations/testy.gif")



# visualise_force_field(vortex_force)
# visualise_force_field(vortex_force_new)
# visualise_force_field(sinusoidal_wave_force)
# visualise_force_field(horizontal_flow_force)
# visualise_force_field(dual_vortex_force)
visualise_force_field(moving_eddies_force)
