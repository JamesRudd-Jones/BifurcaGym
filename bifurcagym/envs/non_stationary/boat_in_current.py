"""
Fluid solver based on Stable Fluids by Jos Stan: "https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf"
"""


import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex
from functools import partial
import jax_cfd.collocated as collocated
import jax_cfd.base as base
import jax_cfd.base.grids as grids
import seaborn as sns


@struct.dataclass
class EnvState(base_env.EnvState):
    x: chex.Array
    y: chex.Array
    fluid_u: chex.Array
    fluid_v: chex.Array
    time: int



"""
Deterministic - There is a constant current 
Homoskedastic - There is a constant current with an applied noise term
Heteroskedastic - There is a current with an applied noise term that is dependent on the state position
Non-Stationary - The current changes over time, perhaps some sinusoidal with time
Chaotic - Uses the fluid solver rather than some arbitrary thing we are using
"""


class BoatInCurrentCSCA(base_env.BaseEnvironment):
    # TODO this may not work well with the generative env, may need to somehow define this better for that to work

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.length: float = 15.0
        self.width: float = 15.0

        self.max_current: float = 0.2

        self.current_noise_scale: float = 0.04

        self.fluid_dt: float = 0.2
        self.fluid_viscosity: float = 1e-6
        self.fluid_iterations: int = 20  # Number of iterations for the linear solver
        self.fluid_force: float = 40.0  # 1.0  # 5.0  # Strength of the external force driving the fluid
        self.current_scaling_factor: float = 0.5  # Scales the effect of the current on the boat

        self.is_deterministic: bool = True
        self.is_heteroskedastic: bool = False
        self.is_non_stationary: bool = False
        self.is_chaotic: bool = True
        self.is_chaotic_new: bool = False

        if self.is_chaotic:
            self.fluid_grid_size: int = 128  # 256
        else:
            self.fluid_grid_size: int = 1  # TODO think this is fine since it still applies the stochasticity?

        self.fluid_grid_size_plot = 128  # 256

        self.grid = grids.Grid((self.fluid_grid_size, self.fluid_grid_size), domain=((0, self.length), (0, self.width)))
        self.density = 1.0
        self.viscosity = 1e-3
        self.max_velocity = 7
        self.cfl_safety_factor = 0.5
        self.dt = base.equations.stable_time_step(self.max_velocity, self.cfl_safety_factor, self.viscosity, self.grid)
        self.inner_steps = 25

        if self.is_non_stationary:
            self.current_func = self.nonstationary_current_func
            self.reset_func = self.nonstationary_reset_func
        elif self.is_chaotic:
            if self.is_chaotic_new:
                self.current_func = self.chaotic_current_func_new
                self.reset_func = self.chaotic_reset_func_new
            else:
                self.current_func = self.chaotic_current_func
                self.reset_func = self.chaotic_reset_func
        else:
            self.current_func = self.stationary_current_func
            self.reset_func = self.stationary_reset_func

        if self.is_deterministic:
            self.noise_func = self.no_noise
            self.is_heteroskedastic = False  # TODO just ensures this for plot/file naming purposes
        else:
            if self.is_heteroskedastic:
                self.noise_func = self.heteroskedastic_noise
            else:
                self.noise_func = self.homoskedastic_noise


        self.goal_state = jnp.array((self.width, self.length))

        self.max_action: float = 1.0
        self.horizon: int = 200

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        uv_interpolated = self._get_current_at_point(state.x, state.y, state.fluid_u, state.fluid_v)
        uv_noise = self.noise_func(state, state.fluid_u, state.fluid_v, key)

        # The solver velocity is in units of (grid_cells / fluid_dt), returned as a displacement vector
        # Scaling factor to make the current's effect noticeable
        x_hat = state.x + action[0] + uv_interpolated[0] * self.current_scaling_factor + uv_noise[0]
        y_hat = state.y + action[1] + uv_interpolated[1] * self.current_scaling_factor + uv_noise[1]

        u, v = self.current_func(state)  # TODO should it be state or new state idk

        new_state = EnvState(x=x_hat, y=y_hat, fluid_u=u, fluid_v=v, time=state.time+1)

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {})

    def _get_current_at_point(self, x, y, u_grid, v_grid):
        grid_x = x / self.width * self.fluid_grid_size
        grid_y = y / self.length * self.fluid_grid_size

        # Coordinates for interpolation
        coords = jnp.array([[grid_y], [grid_x]])

        # Bilinearly interpolate the velocity from the fluid grid
        u_interp = jsp.ndimage.map_coordinates(u_grid, coords, order=1, mode='wrap')[0]
        v_interp = jsp.ndimage.map_coordinates(v_grid, coords, order=1, mode='wrap')[0]

        return jnp.array([u_interp, v_interp])#

    def stationary_current_func(self, state: EnvState) -> Tuple[chex.Array, chex.Array]:
        return state.fluid_u, state.fluid_v

    def stationary_reset_func(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        u_init = -self.max_current * jnp.ones((self.fluid_grid_size, self.fluid_grid_size))
        v_init = self.max_current * jnp.ones((self.fluid_grid_size, self.fluid_grid_size))

        return u_init, v_init

    def nonstationary_current_func(self, state: EnvState) -> Tuple[chex.Array, chex.Array]:
        return state.fluid_u + 0.1 * jnp.sin(state.time), state.fluid_v
        # TODO since time resets at each episode then this isn't fully non-stationary I guess? Although revisiting the same
        # TODO state in the same episode would be different

    def nonstationary_reset_func(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        u_init = -self.max_current * jnp.ones((self.fluid_grid_size, self.fluid_grid_size))
        v_init = self.max_current * jnp.ones((self.fluid_grid_size, self.fluid_grid_size))

        return u_init, v_init

    def chaotic_current_func_new(self, state: EnvState) -> Tuple[chex.Array, chex.Array]:
        u = state.fluid_u
        v = state.fluid_v

        step_fn = base.funcutils.repeated(collocated.equations.semi_implicit_navier_stokes(density=self.density,
                                                                                           viscosity=self.viscosity,
                                                                                           dt=self.dt,
                                                                                           grid=self.grid),
                                          steps=self.inner_steps)

        v0 = collocated.initial_conditions.filtered_velocity_field(jrandom.key(0), self.grid, self.max_velocity)
        # TODO the above feels kind of dodgy but it does work?
        v0[0].array.data = u
        v0[1].array.data = v

        output = jax.device_get(step_fn(v0))

        return output[0].data, output[1].data

    def chaotic_reset_func_new(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        key, _key = jrandom.split(key)

        v0 = collocated.initial_conditions.filtered_velocity_field(_key, self.grid, self.max_velocity)
        # vorticity0 = jaxcfd.finite_differences.curl_2d(v0).data
        # vorticity_hat0 = jnp.fft.rfftn(vorticity0)

        return v0[0].data, v0[1].data

    def chaotic_current_func(self, state: EnvState) -> Tuple[chex.Array, chex.Array]:
        u = state.fluid_u
        v = state.fluid_v

        x_coords, y_coords = jnp.meshgrid(jnp.arange(self.fluid_grid_size),
                                          jnp.arange(self.fluid_grid_size))

        force_u, force_v = self.force_func(x_coords, y_coords, state)

        u += self.fluid_dt * force_u
        v += self.fluid_dt * force_v

        # Standard fluid solver steps (Stable Fluids method)
        u = self._advect(u, u, v)
        v = self._advect(v, u, v)

        u = self._diffuse(u)
        v = self._diffuse(v)
        u, v = self._project(u, v)

        return u, v

    def chaotic_reset_func(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        key, u_key, v_key = jrandom.split(key, 3)
        u_init = jrandom.normal(u_key, (self.fluid_grid_size, self.fluid_grid_size)) * 0.1
        v_init = jrandom.normal(v_key, (self.fluid_grid_size, self.fluid_grid_size)) * 0.1
        # u_init = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))
        # v_init = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))
        u_init, v_init = self._project(u_init, v_init)

        return u_init, v_init

    def force_func(self, x_coords, y_coords, state):
        # A vortex
        centre_x = self.fluid_grid_size / 2
        centre_y = self.fluid_grid_size / 2
        dx, dy = x_coords - centre_x, y_coords - centre_y
        force_u = -dy * self.fluid_force * 1e-4
        force_v = dx * self.fluid_force * 1e-4

        return force_u, force_v

    # def force_func(self, x_world, y_world, state):
    #     num_eddies = 10
    #     amplitude = 10.0
    #     radius = 20
    #     max_speed = 2
    #     key = jrandom.key(42)  # TODO sort out the above at some point
    #
    #     force_u = jnp.zeros_like(x_world, dtype=jnp.float_)
    #     force_v = jnp.zeros_like(y_world, dtype=jnp.float_)
    #
    #     # TODO sort out a proper key here
    #
    #     # Generate fixed random parameters for eddies
    #     key, _key = jrandom.split(key)
    #     initial_pos_x = jrandom.uniform(_key, (num_eddies,)) * self.fluid_grid_size
    #     key, _key = jrandom.split(key)
    #     initial_pos_y = jrandom.uniform(_key, (num_eddies,)) * self.fluid_grid_size
    #
    #     key, _key = jrandom.split(key)
    #     velocities_x = jrandom.uniform(_key, (num_eddies,), minval=-max_speed, maxval=max_speed)
    #     key, _key = jrandom.split(key)
    #     velocities_y = jrandom.uniform(_key, (num_eddies,), minval=-max_speed, maxval=max_speed)
    #     # TODO should this not all go in reset and then we apply the eddies at each step?
    #
    #     # Random spin direction for each eddy
    #     key, _key = jrandom.split(key)
    #     spin = jrandom.choice(_key, jnp.array([-1.0, 1.0]), shape=(num_eddies,))
    #
    #     eddy_params = jnp.stack([initial_pos_x, initial_pos_y, velocities_x, velocities_y, spin], axis=1)
    #
    #     def apply_eddy(carry, params):
    #         force_u_carry, force_v_carry = carry
    #         ix, iy, vx, vy, s = params
    #
    #         # Update eddy center position based on time
    #         center_x = ix + vx * state.time
    #         center_y = iy + vy * state.time
    #         # TODO above will reset when the env resets so is it truly non-stationary?
    #
    #         # Wrap eddies around the domain for continuous flow
    #         center_x = jnp.mod(center_x, self.fluid_grid_size)
    #         center_y = jnp.mod(center_y, self.fluid_grid_size)
    #
    #         dx = x_world - center_x
    #         dy = y_world - center_y
    #
    #         # Vortex force
    #         vortex_u = -dy * 1e-2
    #         vortex_v = dx * 1e-2
    #
    #         # Gaussian falloff for localized effect
    #         distance_sq = dx ** 2 + dy ** 2
    #         falloff = jnp.exp(-distance_sq / (radius ** 2))
    #
    #         force_u_carry += vortex_u * falloff * amplitude * s
    #         force_v_carry += vortex_v * falloff * amplitude * s
    #         return (force_u_carry, force_v_carry), None
    #
    #     (force_u, force_v), _ = jax.lax.scan(apply_eddy, (force_u, force_v), eddy_params)
    #
    #     return force_u, force_v

    def _linear_solve(self, x: chex.Array, b: chex.Array, a: float, c: float) -> chex.Array:
        """
        Solves a linear system using Jacobi iteration with periodic boundary conditions
        Used for diffusion and pressure projection
        """

        def body_fun(_, val):
            x_prev = val
            # Neighbors are found using jnp.roll for periodic boundaries
            neighbors = (jnp.roll(x_prev, 1, axis=0) + jnp.roll(x_prev, -1, axis=0) +
                         jnp.roll(x_prev, 1, axis=1) + jnp.roll(x_prev, -1, axis=1))
            x_new = (b + a * neighbors) / c
            return x_new

        return jax.lax.fori_loop(0, self.fluid_iterations, body_fun, x)

    def _diffuse(self, field: chex.Array) -> chex.Array:
        """Applies fluid diffusion (viscosity)"""
        a = self.fluid_dt * self.fluid_viscosity * self.fluid_grid_size * self.fluid_grid_size
        return self._linear_solve(field, field, a, 1 + 4 * a)

    def _advect(self, field: chex.Array, u: chex.Array, v: chex.Array) -> chex.Array:
        """Moves a quantity 'field' through the velocity field (u, v)"""
        x_coords, y_coords = jnp.meshgrid(jnp.arange(self.fluid_grid_size),
                                          jnp.arange(self.fluid_grid_size))

        # Trace back in time to find the source of the fluid
        back_x = x_coords - (self.fluid_dt * u * self.fluid_grid_size)
        back_y = y_coords - (self.fluid_dt * v * self.fluid_grid_size)

        coords = jnp.stack([back_y, back_x], axis=0)

        # Sample from the original field at the back-traced coordinates
        return jsp.ndimage.map_coordinates(field, coords, order=1, mode='wrap')

    def _project(self, u: chex.Array, v: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Enforces fluid incompressibility"""
        # Calculate divergence using central differences
        div = -0.5 * (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1) +
                      jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0))

        # Solve Poisson equation for pressure
        p = jnp.zeros_like(div)
        p = self._linear_solve(p, -div, 1.0, 4.0)

        # Subtract the pressure gradient from the velocity field
        u_new = u - 0.5 * (jnp.roll(p, -1, axis=1) - jnp.roll(p, 1, axis=1))
        v_new = v - 0.5 * (jnp.roll(p, -1, axis=0) - jnp.roll(p, 1, axis=0))

        return u_new, v_new

    def no_noise(self, state: EnvState, u: chex.Array, v: chex.Array, key: chex.PRNGKey) -> chex.Array:
        return jnp.zeros((2,))

    def homoskedastic_noise(self, state: EnvState, u: chex.Array, v: chex.Array, key: chex.PRNGKey) -> chex.Array:
        x_key, y_key = jrandom.split(key)
        new_u = jrandom.normal(x_key, state.x.shape) * self.current_noise_scale
        new_v = jrandom.normal(x_key, state.y.shape) * self.current_noise_scale

        return jnp.array((new_u, new_v))

    def heteroskedastic_noise(self, state: EnvState, u: chex.Array, v: chex.Array, key: chex.PRNGKey) -> chex.Array:
        x_key, y_key = jrandom.split(key)
        state_depend_x = (state.x / self.width) + 0.5
        state_depend_y = jnp.sin(state.y)

        new_u = jrandom.normal(x_key, state.x.shape) * self.current_noise_scale * state_depend_x
        new_v = jrandom.normal(x_key, state.y.shape) * self.current_noise_scale * state_depend_y

        return jnp.array((new_u, new_v))

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        u_init, v_init = self.reset_func(key)

        state = EnvState(x=jnp.zeros(()),
                         y=jnp.zeros(()),
                         fluid_u=u_init,
                         fluid_v=v_init,
                         time=0)

        return self.get_obs(state), state

    def reward_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> chex.Array:
        return jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y)) - self.goal_state)  # TODO check this is correct

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_action, self.max_action)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array([state.x, state.y])

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], fluid_u=jnp.zeros(()), fluid_v=jnp.zeros(()), time=-1)  # TODO sort this out

    def is_done(self, state: EnvState) -> chex.Array:
        x_bounds = jnp.logical_or(state.x >= self.width, state.x < 0)
        y_bounds = jnp.logical_or(state.y >= self.length, state.y < 0)
        bounds = jnp.logical_or(x_bounds, y_bounds)
        goal_state = jnp.logical_or(state.x - self.goal_state[0] == 0, state.y - self.goal_state[1] == 0)

        done = jnp.logical_or(bounds, goal_state)

        return done

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations/"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import xarray
        import numpy as np

        x_fine = jnp.linspace(0, self.width, self.fluid_grid_size_plot)
        y_fine = jnp.linspace(0, self.length, self.fluid_grid_size_plot)
        X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine)

        coarse_res = 15  # 25
        x_coarse = jnp.linspace(0, self.width, coarse_res)
        y_coarse = jnp.linspace(0, self.length, coarse_res)
        X_coarse, Y_coarse = jnp.meshgrid(x_coarse, y_coarse)

        def current_noise(x, y, u, v, key):
            state = EnvState(x=x, y=y, fluid_u=jnp.zeros(()), fluid_v=jnp.zeros(()), time=-1)
            return self.noise_func(state, u, v, key)

        def vorticity(ds):
            return (ds.v.differentiate('x') - ds.u.differentiate('y')).rename('vorticity')

        if self.is_chaotic:
            ds = xarray.Dataset({'u': (('time', 'x', 'y'), np.asarray(trajectory_state.fluid_u)),
                                 'v': (('time', 'x', 'y'), np.asarray(trajectory_state.fluid_v)),
                                 },
                                coords={'x': self.grid.axes()[0],
                                        'y': self.grid.axes()[1],
                                        'time': self.dt * self.inner_steps * np.arange(trajectory_state.fluid_u.shape[0])}
                                )
            vorticity_ds = vorticity(ds).data

        vmap_current_spatial = jax.vmap(jax.vmap(self._get_current_at_point, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None))
        vmap_current_noise = jax.vmap(jax.vmap(current_noise, in_axes=(0, None, None, None, 0)), in_axes=(None, 0, None, None, 0))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(self.name)
        ax.set_xlim(0, self.width + 1)
        ax.set_ylim(0, self.length + 1)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal', adjustable='box')

        ax.plot(self.goal_state[0], self.goal_state[1], marker='*', markersize=15, color="gold", label="Goal State",
                zorder=4)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

        key = jrandom.key(0)
        key, _key = jrandom.split(key)
        batch_key_fine = jrandom.split(_key, X_fine.shape)
        key, _key = jrandom.split(key)
        batch_key_coarse = jrandom.split(_key, X_coarse.shape)

        def cmap_func_chaotic(index):
            return vorticity_ds[index]

        def cmap_func_non_chaotic(index):
            u_grid, v_grid = trajectory_state.fluid_u[index], trajectory_state.fluid_v[index]
            initial_current_fine = vmap_current_spatial(x_fine, y_fine, u_grid, v_grid)
            initial_current_fine += vmap_current_noise(x_fine, y_fine, u_grid, v_grid, batch_key_fine)
            return jnp.linalg.norm(initial_current_fine, axis=-1)

        if self.is_chaotic:
            cmap_func = cmap_func_chaotic
            cmap_val = sns.cm.icefire
        else:
            cmap_func = cmap_func_non_chaotic
            cmap_val = "plasma"

        initial_u_grid, initial_v_grid = trajectory_state.fluid_u[0], trajectory_state.fluid_v[0]

        initial_mag_fine = cmap_func(0)
        initial_mag_fine = jnp.round(initial_mag_fine, decimals=3)  # stops tiny errors when flow map is constant

        initial_current_coarse = vmap_current_spatial(x_coarse, y_coarse, initial_u_grid, initial_v_grid)
        initial_current_coarse += vmap_current_noise(x_coarse, y_coarse, initial_u_grid, initial_v_grid, batch_key_coarse)
        initial_U, initial_V = initial_current_coarse[:, :, 0], initial_current_coarse[:, :, 1]

        # Setup animation elements
        line, = ax.plot([], [], 'r-', lw=2, label='Agent Trail', zorder=3)
        dot, = ax.plot([], [], color="magenta", marker="o", markersize=10, label='Current State', zorder=5)
        pcm = ax.pcolormesh(X_fine, Y_fine, initial_mag_fine, cmap=cmap_val, shading='auto', zorder=1, alpha=0.7)
        # arrow = ax.quiver(X_coarse, Y_coarse, initial_U, initial_V, color='white', angles='xy', scale_units='xy',
        #                   scale=0.4, width=0.004, zorder=2)
        ax.legend(loc='upper left')

        fig.colorbar(pcm, ax=ax, shrink=0.8, label='Current Magnitude')

        agent_path_x, agent_path_y = [], []

        def update(frame):
            if trajectory_state.time[frame] == 0:
                agent_path_x.clear()
                agent_path_y.clear()

            agent_path_x.append(trajectory_state.x[frame])
            agent_path_y.append(trajectory_state.y[frame])

            line.set_data(agent_path_x, agent_path_y)
            dot.set_data([trajectory_state.x[frame]], [trajectory_state.y[frame]])

            magnitude_fine = cmap_func(frame)

            current_vectors_coarse = vmap_current_spatial(x_coarse, y_coarse, trajectory_state.fluid_u[frame], trajectory_state.fluid_v[frame])
            current_vectors_coarse += vmap_current_noise(x_coarse, y_coarse, trajectory_state.fluid_u[frame], trajectory_state.fluid_v[frame], batch_key_coarse)
            U_coarse = current_vectors_coarse[:, :, 0]
            V_coarse = current_vectors_coarse[:, :, 1]

            pcm.set_array(magnitude_fine.ravel())
            # arrow.set_UVC(U_coarse, V_coarse)

            return line, dot, pcm#, arrow

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(trajectory_state.time),
                                       interval=600,
                                       blit=True)
        if self.is_chaotic_new:
            words = "New"
        elif self.is_chaotic:
            words = "True"
        else:
            words = "False"

        anim.save(f"{file_path}_{self.name}_Dtrmnstc-{self.is_deterministic}_Htrskdstc-{self.is_heteroskedastic}_Nnsttnry-{self.is_non_stationary}_Chaotic-{words}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "BoatInCurrent-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_action, self.max_action, shape=(2,))

    def observation_space(self) -> spaces.Box:
        low = jnp.array([0, 0])
        high = jnp.array([self.width, self.length])
        return spaces.Box(low, high, (2,))


class BoatInCurrentCSDA(BoatInCurrentCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: chex.Array = jnp.array(((0.0, 0.0),
                                                   (-1.0, 0.0),
                                                   (-1.0, -1.0),
                                                   (0.0, -1.0),
                                                   (1.0, 0.0),
                                                   (1.0, 1.0),
                                                   (0.0, 1.0),
                                                   ))

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action.squeeze()]

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
