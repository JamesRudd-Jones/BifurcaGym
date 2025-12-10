""" Fluidic Pinball using D2Q9 Lattice Boltzmann method """

import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex
import cmocean
import numpy as np


@struct.dataclass
class EnvState(base_env.EnvState):
    f: jnp.ndarray
    time: int


class FluidicPinballCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # TODO need to define what is flaot64 or not

        self.nx = 320
        self.ny = 120
        self.reynolds = 100  # Re < 15: Steady, Re ~ 100: Vortex Shedding, Re > 115: Chaotic
        self.tau = 0.6  # Relaxation time (related to viscosity)

        # D2Q9 constraints
        self.w = jnp.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
        self.idxs = jnp.arange(9)
        self.c = jnp.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                            [1, 1], [-1, 1], [-1, -1], [1, -1]])
        # Inverse directions for bounce-back
        self.opp_idxs = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # Triangle formation
        self.centres = jnp.array(((self.nx // 4, self.ny // 2),  # Front cylinder
                                  (self.nx // 4 + 40, self.ny // 2 + 25),  # Top cylinder
                                  (self.nx // 4 + 40, self.ny // 2 - 25)  # Bottom cylinder
                                 ))
        self.radius = 12

        Y, X = jnp.meshgrid(jnp.arange(self.ny), jnp.arange(self.nx), indexing='ij')  # mask for cylinders

        obs_mask_np = np.zeros((self.ny, self.nx), dtype=bool)
        cyl_id_np = np.full((self.ny, self.nx), -1, dtype=np.int32)
        rel_y_np = np.zeros((self.ny, self.nx))
        rel_x_np = np.zeros((self.ny, self.nx))

        for i, (cx, cy) in enumerate(self.centres):
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask = dist <= self.radius
            obs_mask_np |= mask
            cyl_id_np[mask] = i
            rel_y_np[mask] = Y[mask] - cy  # distance for v = omega x r calculation
            rel_x_np[mask] = X[mask] - cx

        self.obs_mask = jnp.array(obs_mask_np)
        self.cyl_id_map = jnp.array(cyl_id_np)
        self.rel_y = jnp.array(rel_y_np)
        self.rel_x = jnp.array(rel_x_np)

        self.max_control = 1

        self.downsample_val = 20

        self.steps_per_env_step = 10
        self.burn_in_steps = 100  # 2000

        self.max_steps = 1000

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)  # Action is angular velocities of pinballs

        # f_next = self.fludic_step(state.f, action)

        def sim_steps(f, unused):
            return self.LBM_step(f, action), None

        f_next, _ = jax.lax.scan(sim_steps, state.f, None, self.steps_per_env_step)
        # TODO run the env for a number of steps for each RL step

        new_state = EnvState(f=f_next, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def LBM_step(self, f, action):
        rho = jnp.sum(f, axis=0)
        u = jnp.einsum('ia,ixy->axy', self.c, f) / rho  # (9, 2).T @ (9, ny, nx) -> (2, ny, nx)

        eu = jnp.einsum('ia,axy->ixy', self.c, u)  # Projection of velocity onto lattice vectors: c_i * u
        u_sq = jnp.sum(u ** 2, axis=0)

        # Calculate Equilibrium (Broadcasting over the 9 channels)
        feq = (self.w[:, None, None] * rho) * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * u_sq)

        f_collided = f - (f - feq) / self.tau  # BGK Collision

        f_post = jnp.where(self.obs_mask[None, :, :], f, f_collided)

        f_bounced = f_post[self.opp_idxs, :, :]  # Bounce-back for obstacles - if maske is true, take opposite f

        # Add Momentum for Rotation (Moving Wall Boundary), for non-obstacle pixels, u_wall is 0.
        # For obstacle pixels, u_wall depends on which cylinder it is (cyl_id_map)

        # Extract omega for each pixel based on cyl_id_map
        # cyl_id_map is -1 for fluid, 0,1,2 for cylinders.
        # jnp.take requires careful indexing. We clamp -1 to 0 temporarily.
        omegas_grid = action[jnp.maximum(self.cyl_id_map, 0)]
        # Where cyl_id_map is -1, set omega to 0
        omegas_grid = jnp.where(self.cyl_id_map == -1, 0.0, omegas_grid)

        # Calculate velocity add-ons: u += -omega * dy, v += omega * dx, sum over 3 cylinders to avoid overlap
        u_wall_x = -omegas_grid * self.rel_y * 0.01
        u_wall_y = omegas_grid * self.rel_x * 0.01

        def add_wall_correction(i, w_i, f_chan):
            ci_dot_uwall = self.c[i, 0] * u_wall_x + self.c[i, 1] * u_wall_y
            correction = 6.0 * w_i * 1.0 * ci_dot_uwall
            return f_chan + correction

        f_corrected = jax.vmap(add_wall_correction)(jnp.arange(9), self.w, f_bounced)

        # Apply mask: Fluid -> f_post, Obstacle -> f_corrected
        f_post = jnp.where(self.obs_mask[None, :, :], f_corrected, f_post)

        def roll_channel(f_i, c_i):
            # Roll x (axis 1), then roll y (axis 0)
            return jnp.roll(jnp.roll(f_i, c_i[0], axis=1), c_i[1], axis=0)

        f_streamed = jax.vmap(roll_channel)(f_post, self.c)

        # 5. Inlet / Outlet
        # Inlet: Fixed velocity (Left wall, x=0)
        # We calculate equilibrium for rho=1, u=(0.1, 0)
        u_inlet = 0.1
        inlet_condition = self.w[:, None] * (1 + 3 * u_inlet)
        f_streamed = f_streamed.at[:, :, 0].set(inlet_condition)
        f_streamed = f_streamed.at[:, :, -1].set(f_streamed[:, :, -2])  #  Outlet: Zero Gradient (Right wall, x=-1) copy from x=-2

        return f_streamed

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)

        f_init = jnp.ones((9, self.ny, self.nx))

        def init_f(w_i, i):
            return 3.0 * w_i * 0.1

        # Add initial velocity term
        f_init = f_init + jax.vmap(init_f, in_axes=(0, 0))(self.w, jnp.arange(9))[:, None, None]

        def burn_in(f, unused):
            return self.LBM_step(f, jnp.zeros(3)), None

        f, _ = jax.lax.scan(burn_in, f_init, None, self.burn_in_steps)

        state = EnvState(f=f, time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        rho = jnp.sum(state_tp1.f, axis=0)
        u = jnp.dot(self.c.T, state_tp1.f.reshape(9, -1)).reshape(2, self.ny, self.nx) / rho

        # Wake slice for drag
        wake_u = u[0, :, self.nx // 2]
        drag_cost = jnp.sum((0.1 - wake_u) ** 2)
        lift_cost = jnp.abs(jnp.sum(u[1, :, self.nx // 2]))
        act_cost = jnp.sum(input_action_t ** 2)

        reward = -(100.0 * drag_cost) - (10.0 * lift_cost) - (0.1 * act_cost)

        # Check stability
        done = (state_tp1.time >= self.max_steps) | jnp.any(jnp.isnan(state_tp1.f)) | (jnp.max(jnp.abs(u)) > 0.5)

        return reward, done

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_control, self.max_control)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        rho = jnp.sum(state.f, axis=0)
        u = jnp.dot(self.c.T, state.f.reshape(9, -1)).reshape(2, self.ny, self.nx) / rho

        # Vorticity calc
        uy, ux = jnp.gradient(u[0])
        vy, vx = jnp.gradient(u[1])
        vorticity = vx - uy

        obs = jnp.stack([u[0], u[1], vorticity])
        obs = obs[:, ::self.downsample_val, ::self.downsample_val]  # Downsample
        flat_obs = jnp.reshape(obs, -1)
        # TODO should we flatten or release as an image?
        return flat_obs

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        raise ValueError("We can't recover state from the partial obs")
        return EnvState(f=obs[0], time=-1)  # TODO this is not possible

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        x_fine = jnp.arange(0, self.nx, 1)
        y_fine = jnp.arange(0, self.ny, 1)
        X, Y = jnp.meshgrid(x_fine, y_fine)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(self.name)
        # ax.set_xlim(float(self.x_bounds[0]), float(self.x_bounds[1]))
        # ax.set_ylim(float(self.y_bounds[0]), float(self.y_bounds[1]))
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal', adjustable='box')

        cmap_val = cmocean.cm.curl

        def get_plot_obs(f):
            rho = jnp.sum(f, axis=0)
            u = jnp.dot(self.c.T, f.reshape(9, -1)).reshape(2, self.ny, self.nx) / rho

            # Vorticity calc
            uy, ux = jnp.gradient(u[0])
            vy, vx = jnp.gradient(u[1])
            vorticity = vx - uy

            return u[0], u[1], vorticity

        u_0, u_1, vorticity = get_plot_obs(trajectory_state.f[0])

        # Setup animation elements
        dot, = ax.plot([], [], color="magenta", marker="o", markersize=10, label='Pinball', zorder=5)
        pcm = ax.pcolormesh(X, Y, vorticity, cmap=cmap_val, shading='auto', zorder=1, alpha=0.7)
        # arrow = ax.quiver(X_coarse, Y_coarse, initial_U_coarse, initial_V_coarse, color='black', angles='xy', scale_units='xy',
        #                   scale=3, width=0.004, zorder=2)
        ax.legend(loc='upper left')

        fig.colorbar(pcm, ax=ax, shrink=0.4, label='Current Magnitude')

        def update(frame):
            # dot.set_data([trajectory_state.x[frame]], [trajectory_state.y[frame]])

            u_0, u_1, vorticity = get_plot_obs(trajectory_state.f[frame])
            pcm.set_array(vorticity.ravel())

            # arrow.set_UVC(U_coarse, V_coarse)

            return dot, pcm  # , arrow

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(trajectory_state.time),
                                       interval=200,
                                       blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "FluidicPinball-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, shape=(3,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(-1.4, 1.4, (3 * self.nx // self.downsample_val * self.ny // self.downsample_val,), dtype=jnp.float64)


class FluidicPinballCSDA(FluidicPinballCSCA):
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
        return self.action_array[action.squeeze()] * self.max_control

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
