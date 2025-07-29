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


@struct.dataclass
class EnvState(base_env.EnvState):
    x: chex.Array
    y: chex.Array
    fluid_u: chex.Array
    fluid_v: chex.Array
    time: int
    key: chex.PRNGKey



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

        self.current_noise_scale: float = 0.5

        self.fluid_grid_size: int = 10
        self.fluid_dt: float = 0.2
        self.fluid_viscosity: float = 1e-6
        self.fluid_iterations: int = 20  # Number of iterations for the linear solver
        self.fluid_force: float = 40.0  # 1.0  # 5.0  # Strength of the external force driving the fluid
        self.current_scaling_factor: float = 1.2  # Scales the effect of the current on the boat

        self.is_homoskedastic: bool = False
        self.is_heteroskedastic: bool = False
        self.is_non_stationary: bool = False
        self.is_chaotic: bool = True

        if self.is_non_stationary:
            self.current_func = self.nonstationary_current_func
        elif self.is_chaotic:
            self.current_func = self.chaotic_current_func
        else:
            self.current_func = self.stationary_current_func

        if self.is_homoskedastic:
            self.noise_func = self.homoskedastic_noise
        elif self.is_heteroskedastic:
            self.noise_func = self.heteroskedastic_noise
        else:
            self.noise_func = self.no_noise

        self.goal_state = jnp.array((self.width, self.length))

        self.max_action: float = 1.0
        self.horizon: int = 200

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        u, v = self.current_func(state)
        u, v = self.noise_func(state, u, v, key)

        grid_x = state.x / self.width * self.fluid_grid_size
        grid_y = state.y / self.length * self.fluid_grid_size

        # Create coordinate array for interpolation
        coords = jnp.array([[grid_y], [grid_x]])
        # TODO should this be the other way around?

        # Bilinearly interpolate the velocity from the fluid grid
        u_interpolated = jsp.ndimage.map_coordinates(state.fluid_u, coords, order=1, mode='wrap')[0]
        v_interpolated = jsp.ndimage.map_coordinates(state.fluid_v, coords, order=1, mode='wrap')[0]

        # The solver velocity is in units of (grid_cells / fluid_dt).
        # We return this as a displacement vector for the agent's timestep.
        # A scaling factor is used to make the current's effect noticeable.
        x_hat = state.x + action[0] + u_interpolated * self.current_scaling_factor
        y_hat = state.y + action[1] + v_interpolated* self.current_scaling_factor

        new_state = EnvState(x=x_hat, y=y_hat, fluid_u=u, fluid_v=v, time=state.time + 1, key=key)

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,  # TODO check reward is the correct way around
                self.is_done(new_state),
                {})

    def _get_current_at_point(self, x, y, u_grid, v_grid):
        grid_x = x / self.width * self.fluid_grid_size
        grid_y = y / self.length * self.fluid_grid_size
        coords = jnp.array([[grid_y], [grid_x]])
        u_interp = jsp.ndimage.map_coordinates(u_grid, coords, order=1, mode='wrap')[0]
        v_interp = jsp.ndimage.map_coordinates(v_grid, coords, order=1, mode='wrap')[0]
        return jnp.array([u_interp, v_interp])

    def stationary_current_func(self, state: EnvState) -> chex.Array:
        return jnp.array((-self.max_current, self.max_current))

    def nonstationary_current_func(self, state: EnvState) -> chex.Array:
        return jnp.array((-self.max_current + 0.1 * jnp.sin(state.time),
                          self.max_current,
                          ))

    def chaotic_current_func(self, state: EnvState) -> Tuple[chex.Array, chex.Array]:
        u = state.fluid_u
        v = state.fluid_v

        # Example: A circular force field
        x_coords, y_coords = jnp.meshgrid(jnp.arange(self.fluid_grid_size * self.width),
                                          jnp.arange(self.fluid_grid_size * self.length))
        centre_x = (self.fluid_grid_size * self.width) / 2
        centre_y = (self.fluid_grid_size * self.length) / 2
        dx, dy = x_coords - centre_x, y_coords - centre_y
        force_u = -dy * self.fluid_force * 1e-4  # creates a vortex
        # force_u = dy * self.fluid_force * 1e-4
        force_v = dx * self.fluid_force * 1e-4

        u += self.fluid_dt * force_u
        v += self.fluid_dt * force_v

        # Standard fluid solver steps (Stable Fluids method)
        u = self._diffuse(u)
        v = self._diffuse(v)

        u = self._advect(u, u, v)
        v = self._advect(v, u, v)
        u, v = self._project(u, v)

        return u, v

    def no_noise(self, state: EnvState, u: chex.Array, v: chex.Array, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        return u, v

    def homoskedastic_noise(self, state: EnvState, u: chex.Array, v: chex.Array, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        x_key, y_key = jrandom.split(key)
        return jnp.array((current[0] + jrandom.normal(x_key) * self.current_noise_scale,
                          current[1] + jrandom.normal(y_key) * self.current_noise_scale,
                          ))

    def heteroskedastic_noise(self, state: EnvState, u: chex.Array, v: chex.Array, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        x_key, y_key = jrandom.split(key)
        state_depend_x = (state.x / self.width) + 0.5
        state_depend_y = (state.y / self.length)
        return jnp.array((current[0] + (jrandom.normal(x_key) * self.current_noise_scale * state_depend_x),
                          current[1] + (jrandom.normal(y_key) * self.current_noise_scale * state_depend_y),
                          ))

    def _linear_solve(self, x: chex.Array, b: chex.Array, a: float, c: float) -> chex.Array:
        """
        Solves a linear system using Jacobi iteration with periodic boundary conditions.
        Used for diffusion and pressure projection.
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
        a = self.fluid_dt * self.fluid_viscosity * (self.fluid_grid_size * self.width) * (self.fluid_grid_size * self.length)
        return self._linear_solve(field, field, a, 1 + 4 * a)

    def _advect(self, field: chex.Array, u: chex.Array, v: chex.Array) -> chex.Array:
        """Moves a quantity 'field' through the velocity field (u, v)."""
        x_coords, y_coords = jnp.meshgrid(jnp.arange(self.width * self.fluid_grid_size),
                                          jnp.arange(self.length * self.fluid_grid_size))

        # Trace back in time to find the source of the fluid
        back_x = x_coords - (self.fluid_dt * u * (self.width * self.fluid_grid_size))
        back_y = y_coords - (self.fluid_dt * v * (self.length * self.fluid_grid_size))

        coords = jnp.stack([back_y, back_x], axis=0)

        # Sample from the original field at the back-traced coordinates
        return jsp.ndimage.map_coordinates(field, coords, order=1, mode='wrap')

    def _project(self, u: chex.Array, v: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Enforces fluid incompressibility."""
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

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # # TODO below is for normal start
        # u_init = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))
        # v_init = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))

        # TODO below is for fluid solver
        key, u_key, v_key = jrandom.split(key, 3)
        u_init = jrandom.normal(u_key, (int(self.fluid_grid_size * self.width), int(self.fluid_grid_size * self.length))) * 0.1
        v_init = jrandom.normal(v_key, (int(self.fluid_grid_size * self.width), int(self.fluid_grid_size * self.length))) * 0.1
        u_init, v_init = self._project(u_init, v_init)

        state = EnvState(x=jnp.zeros(()),
                         y=jnp.zeros(()),
                         fluid_u=u_init,
                         fluid_v=v_init,
                         time=0,
                         key=key)

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
        return EnvState(x=obs[0], y=obs[1], time=-1, key=key)  # TODO sort this out

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

        x_fine = jnp.linspace(0, self.width, int(self.fluid_grid_size * self.width))
        y_fine = jnp.linspace(0, self.length, int(self.fluid_grid_size * self.length))
        X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine)

        coarse_res = 15  # 25
        x_coarse = jnp.linspace(0, self.width, coarse_res)
        y_coarse = jnp.linspace(0, self.length, coarse_res)
        X_coarse, Y_coarse = jnp.meshgrid(x_coarse, y_coarse)

        vmap_current_spatial = jax.vmap(jax.vmap(self._get_current_at_point, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None))

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

        initial_u_grid, initial_v_grid = trajectory_state.fluid_u[0], trajectory_state.fluid_v[0]
        initial_current_fine = vmap_current_spatial(x_fine, y_fine, initial_u_grid, initial_v_grid)
        # initial_current_fine = vmap_current(x_fine, y_fine, trajectory_state.time[0], trajectory_state.key[0])
        initial_mag_fine = jnp.linalg.norm(initial_current_fine, axis=-1)
        initial_current_coarse = vmap_current_spatial(x_coarse, y_coarse, initial_u_grid, initial_v_grid)
        # initial_current_coarse = vmap_current(x_coarse, y_coarse, trajectory_state.time[0], trajectory_state.key[0])
        initial_U, initial_V = initial_current_coarse[:, :, 0], initial_current_coarse[:, :, 1]

        # Setup animation elements
        line, = ax.plot([], [], 'r-', lw=2, label='Agent Trail', zorder=3)
        dot, = ax.plot([], [], color="magenta", marker="o", markersize=10, label='Current State', zorder=5)
        pcm = ax.pcolormesh(X_fine, Y_fine, initial_mag_fine, cmap='viridis', shading='auto', zorder=1, alpha=0.7)
        arrow = ax.quiver(X_coarse, Y_coarse, initial_U, initial_V, color='white', angles='xy', scale_units='xy',
                          scale=0.4, width=0.004, zorder=2)
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

            current_vectors_fine = vmap_current_spatial(x_fine, y_fine, trajectory_state.fluid_u[frame], trajectory_state.fluid_v[frame])
            # current_vectors_fine = vmap_current(x_fine, y_fine, trajectory_state.time[frame], trajectory_state.key[frame])
            magnitude_fine = jnp.linalg.norm(current_vectors_fine, axis=-1)

            current_vectors_coarse = vmap_current_spatial(x_coarse, y_coarse, trajectory_state.fluid_u[frame], trajectory_state.fluid_v[frame])
            # current_vectors_coarse = vmap_current(x_coarse, y_coarse, trajectory_state.time[frame], trajectory_state.key[frame])
            U_coarse = current_vectors_coarse[:, :, 0]
            V_coarse = current_vectors_coarse[:, :, 1]

            pcm.set_array(magnitude_fine.ravel())
            arrow.set_UVC(U_coarse, V_coarse)

            return line, dot, pcm, arrow

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(trajectory_state.time),
                                       interval=600,
                                       blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
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



