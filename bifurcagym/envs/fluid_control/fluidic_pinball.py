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


jax.config.update("jax_enable_x64", True)


@struct.dataclass
class EnvState(base_env.EnvState):
    f: jnp.ndarray
    drag: jnp.ndarray
    lift: jnp.ndarray
    time: int


class FluidicPinballCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # TODO need to define what is float64 or not

        self.nx = 300
        self.ny = 150
        self.reynolds = 100.0  # Re < 15: Steady, Re ~ 100: Vortex Shedding, Re > 115: Chaotic
        self.tau = 0.6  # Relaxation time (related to viscosity)

        # D2Q9 constraints
        self.w = jnp.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=jnp.float64)
        self.idxs = jnp.arange(9)
        self.c = jnp.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                            [1, 1], [-1, 1], [-1, -1], [1, -1]])
        # Inverse directions for bounce-back
        self.opp_idxs = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # Triangle formation
        self.centres = jnp.array(((self.nx // 4, self.ny // 2),  # Front cylinder
                                  (4 * self.nx // 10, self.ny // 2 + 25),  # Top cylinder
                                  (4 * self.nx // 10, self.ny // 2 - 25)  # Bottom cylinder
                                 ))
        self.radius = 10

        self.u_inlet = 0.1  # Inlet velocity
        self.cs2 = 1 / 3  # Speed of sound squared
        self.nu = self.u_inlet * (self.ny / 6.0) / self.reynolds  # Kinematic viscosity (derived from Re)
        self.tau = 3 * self.nu + 0.5  # Relaxation time

        self.X, self.Y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny))  # mask for cylinders

        mask = jnp.zeros((self.ny, self.nx), dtype=bool)
        for cx, cy in self.centres:
            dist = jnp.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
            mask = mask | (dist <= self.radius)

        self.mask_solid = mask

        self.max_control = 1

        self.downsample_val = 20

        self.steps_per_env_step = 20
        self.burn_in_steps = 20#00

        self.max_steps = 1000

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)  # Action is angular velocities of pinballs

        # f_next = self.LBM_step(state.f, action)

        def sim_steps(runner_state, unused):
            f, _, _ = runner_state
            return self.LBM_step(f, action), None

        (f_next, drag, lift), _ = jax.lax.scan(sim_steps, (state.f, jnp.zeros(()), jnp.zeros(())), None, self.steps_per_env_step)
        # TODO run the env for a number of steps for each RL step

        new_state = EnvState(f=f_next, drag=drag, lift=lift, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def LBM_step(self, f, action):
        rho_YX1 = jnp.sum(f, axis=-1, keepdims=True)
        u = jnp.dot(f, self.c) / rho_YX1

        feq = self._get_equilibrium(rho_YX1, u)
        f_post_coll = f - (f - feq) / self.tau

        f_boundary = self._boundary_conditions(f_post_coll, action)

        # def _stream(f):
        #     f_streamed = f
        #     for i in range(1, 9):
        #         f_streamed = f_streamed.at[:, :, i].set(jnp.roll(f[:, :, i], shift=self.c[i], axis=(1, 0)))
        #     return f_streamed
        #
        # f_next = _stream(f_boundary)

        def stream_step(f, idx):
            f = f.at[:, :, idx].set(jnp.roll(f_boundary[:, :, idx], shift=self.c[idx], axis=(1, 0)))
            return f, None

        f_next, _ = jax.lax.scan(stream_step, f_boundary, jnp.arange(1, 9, 1), 8)

        drag, lift = self._calculate_forces(f_post_coll, f_boundary)

        return f_next, drag, lift

    def _get_equilibrium(self, rho, u):
        # Projects macroscopic velocity onto discrete directions
        cu = jnp.einsum('...d,qd->...q', u, self.c)
        usqr = jnp.einsum('...d,...d->...', u, u)
        cu3 = 3.0 * cu
        usqr = usqr[..., jnp.newaxis]
        feq = self.w * rho * (1.0 + cu3 + 0.5 * cu3 ** 2 - 1.5 * usqr)
        return feq

    def _boundary_conditions(self, f, action):
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
            uw_x = -omega * (self.Y - cy)
            uw_y = omega * (self.X - cx)

            # We only care about this velocity AT the solid boundary pixels
            # Standard bounce-back: f_in(direction) -> f_out(opposite)
            # Moving wall correction: f_out = f_in - 2*w*rho*(u_wall . c)/cs2

            # Calculate momentum correction term
            # Dot product of Wall Velocity and Direction Vectors
            # We do this for all 9 directions at once
            wall_dot_c = (uw_x[..., None] * self.c[:, 0]) + (uw_y[..., None] * self.c[:, 1])
            correction = 2.0 * self.w * 1.0 * (wall_dot_c / self.cs2)  # approx rho=1.0 at wall

            # Apply specifically where the mask is solid
            # Note: In a full code, we optimize to only do boundary nodes.
            # Here we mask the whole grid for simplicity.
            f_bounced = f[:, :, self.opp_idxs]  # The standard bounce
            f_rotated = f_bounced - correction

            # Update ONLY the solid pixels for this cylinder
            # Create a specific mask for this cylinder to avoid overlap issues
            dist = jnp.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
            cyl_mask = (dist <= 10.0)  # Match radius above

            f_new = jnp.where(cyl_mask[..., None], f_rotated, f_new)

            return f_new, None

        scan_input = jnp.concatenate((self.centres, jnp.expand_dims(action, axis=-1)), axis=-1)
        f_new, _ = jax.lax.scan(cylinder_step, f_new, scan_input, 3)

        rho_inlet = 1.0
        u_vec_inlet = jnp.zeros((self.ny, 1, 2)).at[..., 0].set(self.u_inlet)
        feq_inlet = self._get_equilibrium(rho_inlet, u_vec_inlet)
        f_new = f_new.at[:, 0, :].set(feq_inlet[:, 0, :])  # Force the left column to be equilibrium
        f_new = f_new.at[:, -1, :].set(f_new[:, -2, :])  # Zero-gradient (copy second to last column to last column)

        return f_new

    def _calculate_forces(self, f_pre_collision, f_post_collision):
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
        f_solid = jnp.where(self.mask_solid[..., None], f_post_collision - f_pre_collision, 0.0)

        # Project onto X and Y
        # Sum over all directions and all solid pixels
        fx = jnp.sum(f_solid * self.c[:, 0])
        fy = jnp.sum(f_solid * self.c[:, 1])

        return fx, fy

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        f_init = self._get_equilibrium(jnp.ones((self.ny, self.nx, 1)), jnp.zeros((self.ny, self.nx, 2)))

        def burn_in(runner_state, unused):
            f, _, _ = runner_state
            return self.LBM_step(f, jnp.zeros(3)), None

        (f, drag, lift), _ = jax.lax.scan(burn_in, (f_init, jnp.zeros(()), jnp.zeros(())), None, self.burn_in_steps)

        state = EnvState(f=f, drag=drag, lift=lift, time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        rho = jnp.sum(state_tp1.f, axis=-1, keepdims=True)
        u = jnp.dot(state_tp1.f, self.c) / rho

        reward = -(state_tp1.drag + 0.1 * jnp.abs(state_tp1.lift))

        # Check stability
        done = (state_tp1.time >= self.max_steps) | jnp.any(jnp.isnan(state_tp1.f)) | (jnp.max(jnp.abs(u)) > 0.5)

        return reward, done

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_control, self.max_control)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        rho = jnp.sum(state.f, axis=-1, keepdims=True)
        u = jnp.dot(state.f, self.c) / rho

        # Vorticity calc
        uy, ux = jnp.gradient(u[:, :, 0])
        vy, vx = jnp.gradient(u[:, :, 1])
        vorticity = vx - uy

        obs = jnp.stack([u[:, :, 0], u[:, :, 1], vorticity])
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
            rho = jnp.sum(f, axis=-1, keepdims=True)
            u = jnp.dot(f, self.c) / rho

            # Vorticity calc
            uy, ux = jnp.gradient(u[:, :, 0])
            vy, vx = jnp.gradient(u[:, :, 1])
            vorticity = vx - uy

            return u[:, :, 0], u[:, :, 1], vorticity

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
