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
    drag: jnp.ndarray
    lift: jnp.ndarray


@struct.dataclass
class EnvParams:
    values = jnp.array([-1, 0, 1])
    action_array: chex.Array = struct.field(False, default=jnp.array(jnp.meshgrid(values, values, values)).T.reshape(-1, 3))
    burn_in_steps: int = struct.field(False, default=2000)
    downsample_val: int = struct.field(False, default=20)
    max_steps_in_ep: int = struct.field(False, default=1000)
    steps_per_env_step: int = struct.field(False, default=20)

    nx: int = struct.field(False, default=300)
    ny: int = struct.field(False, default=150)
    reynolds: float = 105.0  # Re < 15: Steady, Re ~ 100: Vortex Shedding, Re > 115: Chaotic
    tau: float = 0.6  # Relaxation time (related to viscosity)

    # D2Q9 constraints
    w: chex.Array = struct.field(False, default=jnp.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=jnp.float64))
    idxs: chex.Array = struct.field(False, default=jnp.arange(9))
    c: chex.Array = struct.field(False, default=jnp.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                                                                        [1, 1], [-1, 1], [-1, -1], [1, -1]]))
    # Inverse directions for bounce-back
    opp_idxs: chex.Array = struct.field(False, default=jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6]))

    radius: int = struct.field(False, default=10)

    u_inlet: float = 0.1  # Inlet velocity
    cs2: float = 1 / 3  # Speed of sound squared

    max_control: float = 1.0
    maximum_max_control: float = struct.field(False, default=1.0)  # maximum to ensure correct scaling

    @property
    def centres(self) -> chex.Array:
        # Triangle formation
        return jnp.array(((self.nx // 4, self.ny // 2),  # Front cylinder
                          (4 * self.nx // 10, self.ny // 2 + 15),  # Top cylinder
                          (4 * self.nx // 10, self.ny // 2 - 15)  # Bottom cylinder
                          ))

    @property
    def nu(self) -> float:
        return self.u_inlet * (self.ny / 6.0) / self.reynolds  # Kinematic viscosity (derived from Re)

    @property
    def tau(self) -> float:
        return 3 * self.nu + 0.5  # Relaxation time

    @property
    def mask_solid(self) -> chex.Array:
        mask = jnp.zeros((self.ny, self.nx), dtype=bool)
        for cx, cy in self.centres:
            dist = jnp.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
            mask = mask | (dist <= self.radius)

        return mask

    @property
    def _meshgrid(self):
        return jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny))  # mask for cylinders

    @property
    def X(self):
        return self._meshgrid[0]

    @property
    def Y(self):
        return self._meshgrid[1]


class FluidicPinballCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(self,
                 input_action: chex.Numeric,
                 state: EnvState,
                 params: EnvParams,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action, params)  # Action is angular velocities of pinballs

        # f_next = self.LBM_step(state.f, action)

        def sim_steps(runner_state, unused):
            f, _, _ = runner_state
            return self.LBM_step(f, action, params), None

        (f_next, drag, lift), _ = jax.lax.scan(sim_steps, (state.f, jnp.zeros(()), jnp.zeros(())), None, params.steps_per_env_step)
        # TODO run the env for a number of steps for each RL step

        new_state = EnvState(f=f_next, drag=drag, lift=lift, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def LBM_step(self, f, action, params: EnvParams):
        rho_YX1 = jnp.sum(f, axis=-1, keepdims=True)
        u = jnp.dot(f, params.c) / rho_YX1

        feq = self._get_equilibrium(rho_YX1, u, params)
        f_post_coll = f - (f - feq) / params.tau

        f_boundary = self._boundary_conditions(f_post_coll, action, params)

        f0 = f_boundary[:, :, 0]
        f1 = jnp.roll(f_boundary[:, :, 1], (1, 0), axis=(1, 0))
        f2 = jnp.roll(f_boundary[:, :, 2], (0, 1), axis=(1, 0))
        f3 = jnp.roll(f_boundary[:, :, 3], (-1, 0), axis=(1, 0))
        f4 = jnp.roll(f_boundary[:, :, 4], (0, -1), axis=(1, 0))
        f5 = jnp.roll(f_boundary[:, :, 5], (1, 1), axis=(1, 0))
        f6 = jnp.roll(f_boundary[:, :, 6], (-1, 1), axis=(1, 0))
        f7 = jnp.roll(f_boundary[:, :, 7], (-1, -1), axis=(1, 0))
        f8 = jnp.roll(f_boundary[:, :, 8], (1, -1), axis=(1, 0))

        f_next = jnp.stack((f0, f1, f2, f3, f4, f5, f6, f7, f8), axis=2)

        drag, lift = self._calculate_forces(f_post_coll, f_boundary, params)

        return f_next, drag, lift

    def _get_equilibrium(self, rho, u, params: EnvParams):
        # Projects macroscopic velocity onto discrete directions
        cu = jnp.dot(u, params.c.T)
        usqr = jnp.sum(u ** 2, axis=-1, keepdims=True)
        feq = params.w * rho * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * usqr)
        return feq

    def _boundary_conditions(self, f, action, params):
        """
        Applies:
        1. Inlet (Left) - Constant Velocity
        2. Outlet (Right) - Outflow
        3. Cylinders - Rotating Bounce-Back (The 'Action')
        """

        # Cylinder boune back steps
        # centres: (Ncyl, 2)
        # action:  (Ncyl,)
        cx = params.centres[:, 0][:, None, None]  # (Ncyl, 1, 1)
        cy = params.centres[:, 1][:, None, None]
        omega = action[:, None, None]

        # Wall velocity: u = omega x r
        uw_x = -omega * (params.Y - cy)
        uw_y = omega * (params.X - cx)

        # Wall velocity dot lattice directions
        # Result: (Ncyl, ny, nx, 9)
        wall_dot_c = (uw_x[..., None] * params.c[:, 0] + uw_y[..., None] * params.c[:, 1])

        correction = 2.0 * params.w * (wall_dot_c / params.cs2)

        # Standard bounce-back
        f_bounced = f[..., params.opp_idxs]  # (ny, nx, 9)

        # Rotating bounce-back per cylinder
        f_rotated = f_bounced[None, ...] - correction

        # Cylinder masks (use squared distance)
        r2 = (params.X - cx) ** 2 + (params.Y - cy) ** 2
        cyl_mask = r2 <= (10.0 ** 2)  # (Ncyl, ny, nx)

        # Combine cylinders safely (no overwrite ambiguity)
        cyl_mask_any = jnp.any(cyl_mask, axis=0)

        # If overlapping cylinders exist, last one wins (consistent with scan)
        idx = jnp.argmax(cyl_mask, axis=0)
        f_cyl = jnp.take_along_axis(f_rotated, idx[None, ..., None], axis=0)[0]

        f_new = jnp.where(cyl_mask_any[..., None], f_cyl, f)

        rho_inlet = 1.0
        u_inlet_vec = jnp.zeros((params.ny, 1, 2)).at[..., 0].set(params.u_inlet)
        feq_inlet = self._get_equilibrium(rho_inlet, u_inlet_vec, params)
        f_new = f_new.at[:, 0, :].set(feq_inlet[:, 0, :])  # Inlet (Left boundary): Constant velocity
        f_new = f_new.at[:, -1, :].set(f_new[:, -2, :])  # Outlet (Right boundary): Zero-gradient

        return f_new

    def _calculate_forces(self, f_pre_collision, f_post_collision, params: EnvParams):
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
                           params.mask_solid.astype(delta_f.dtype), # Ensure mask is float
                           delta_f,
                           params.c)

        return force

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        f_init = self._get_equilibrium(jnp.ones((params.ny, params.nx, 1)), jnp.zeros((params.ny, params.nx, 2)), params)

        def burn_in(runner_state, unused):
            f, _, _ = runner_state
            return self.LBM_step(f, jnp.zeros(3), params), None

        (f, drag, lift), _ = jax.lax.scan(burn_in, (f_init, jnp.zeros(()), jnp.zeros(())), None, params.burn_in_steps)

        state = EnvState(f=f, drag=drag, lift=lift, time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        rho = jnp.sum(state_tp1.f, axis=-1, keepdims=True)
        u = jnp.dot(state_tp1.f, params.c) / rho

        reward = -(state_tp1.drag + 0.1 * jnp.abs(state_tp1.lift))

        # Check stability
        done = (state_tp1.time >= params.max_steps_in_ep) | jnp.any(jnp.isnan(state_tp1.f)) | (jnp.max(jnp.abs(u)) > 0.5)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control)

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
        # TODO how to use params again as not mentioned here

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        raise ValueError("We can't recover state from the partial obs")
        return EnvState(f=obs[0], time=-1)  # TODO this is not possible

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        fs = np.asarray(trajectory_state.f)
        cs = np.asarray(params.c)

        x_fine = np.arange(0, params.nx, 1)
        y_fine = np.arange(0, params.ny, 1)
        X, Y = np.meshgrid(x_fine, y_fine)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(self.name)
        # ax.set_xlim(float(self.x_bounds[0]), float(self.x_bounds[1]))
        # ax.set_ylim(float(self.y_bounds[0]), float(self.y_bounds[1]))
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal', adjustable='box')

        cmap_val = cmocean.cm.curl

        def get_plot_obs(f, c):
            rho = np.sum(f, axis=-1, keepdims=True)
            u = np.dot(f, c) / rho

            # Vorticity calc
            uy, ux = np.gradient(u[:, :, 0])
            vy, vx = np.gradient(u[:, :, 1])
            vorticity = vx - uy

            return u[:, :, 0], u[:, :, 1], vorticity

        u_0, u_1, vorticity = get_plot_obs(fs[0], cs[0])

        # Setup animation elements
        dot, = ax.plot([], [], color="magenta", marker="o", markersize=10, label='Pinball', zorder=5)
        pcm = ax.pcolormesh(X, Y, vorticity, cmap=cmap_val, shading='auto', zorder=1, alpha=0.7)
        # arrow = ax.quiver(X_coarse, Y_coarse, initial_U_coarse, initial_V_coarse, color='black', angles='xy', scale_units='xy',
        #                   scale=3, width=0.004, zorder=2)
        ax.legend(loc='upper left')

        fig.colorbar(pcm, ax=ax, shrink=0.4, label='Current Magnitude')

        def update(frame):
            # dot.set_data([trajectory_state.x[frame]], [trajectory_state.y[frame]])

            u_0, u_1, vorticity = get_plot_obs(fs[frame], cs[frame])
            pcm.set_array(vorticity.ravel())

            # arrow.set_UVC(U_coarse, V_coarse)

            return dot, pcm  # , arrow

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(fs),
                                       interval=200,
                                       blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "FluidicPinball-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(3,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-1.4, 1.4, (3 * params.nx // params.downsample_val * params.ny // params.downsample_val,), dtype=jnp.float64)


class FluidicPinballCSDA(FluidicPinballCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return params.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(params.action_array))
