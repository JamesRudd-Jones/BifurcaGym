"""
Adaption of the AYS environment into JAX, enabling full vectorisation
Original: https://github.com/fstrnad/pyDRLinWESM
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Any, Dict, Tuple
import chex
from flax import struct
from bifurcagym.envs import base_env, utils
from bifurcagym import spaces


@struct.dataclass
class EnvState(base_env.EnvState):
    ays: jnp.ndarray


@struct.dataclass
class EnvParams:
    A_PB: float = struct.field(False, default=0.5897)  # Planetary boundary
    Y_PB: float = struct.field(False, default=0.3636)  # Social foundation
    S_LIMIT: float = struct.field(False, default=0.0)

    random_start_limit: float = 0.05

    A_offset: float = 600

    tau_A: float = 50.0
    tau_S: float = 50.0
    beta: float = 0.03
    eps: float = 147.0
    rho: float = 2.0
    sigma: float = 4e12
    phi: float = 4.7e10

    # Fixed points for termination conditions
    green_fp: chex.Array = struct.field(False, default=jnp.array([0.0, 1.0, 1.0]))
    black_fp: chex.Array = struct.field(False, default=jnp.array([0.6, 0.4, 0.0]))
    final_radius: float = struct.field(False, default=0.05)

    @property
    def theta(self) -> float:
        return self.beta / (950 - self.A_offset)


class AYSIAMCSDA(base_env.BaseEnvironment):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = 1
        self.horizon: int = 25
        self.max_steps_in_ep: int = 600
        self.substeps: int = 10

        self.action_array: chex.Array = jnp.array(((1.0, 0.0),
                                                   (0.5, 0.0),
                                                   (1.0, 1.0),
                                                   (0.5, 1.0)
                                                   ))
        # AYS Action Mapping: 0=NOTHING, 1=LG, 2=ET, 3=LG+ET
        # beta is 0.03 and then it is 0.015 for LG
        # sigma is 4e12 and then it is 4e12 * 0.5 ** (1 / rho) - important since rho could change so it is defined as a param and not hardcoded

        self.requires_float64: bool = False

        self.defined_param_start: bool = False
        self.evaluating: bool = False

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(self,
                 input_action: chex.Numeric,
                 state: EnvState,
                 params: EnvParams,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action, params)

        new_ays = utils.integrate_ode(self._ays_rhs, state.time * self.dt, state.ays, action, self.dt, self.substeps, params)

        new_state = EnvState(ays=new_ays,
                             time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {"discount": self.discount(done)})

    def _ays_rhs(self, t: float, ays: chex.Array, action: chex.Array, params: EnvParams) -> chex.Array:
        A_mid = 240.0  # TODO or is it 250, need to check?
        Y_mid = 7e13
        S_mid = 5e11

        ays_inv = 1.0 - ays
        inv_s_rho = ays_inv[2] ** params.rho

        sigma = 4e12 * 0.5 ** (action[1] / params.rho)

        A_val = A_mid * (ays[0] / ays_inv[0])
        Y_val = Y_mid * (ays[1] / ays_inv[1])

        K = inv_s_rho / (inv_s_rho + (S_mid * ays[2] / sigma) ** params.rho)

        adot = K / (params.phi * params.eps * A_mid) * ays_inv[0] * ays_inv[0] * Y_val - ays[0] * ays_inv[0] / params.tau_A
        ydot = ays[1] * ays_inv[1] * ((params.beta * action[0]) - params.theta * A_val)
        sdot = (1.0 - K) * ays_inv[2] * ays_inv[2] * Y_val / (params.eps * S_mid) - ays[2] * ays_inv[2] / params.tau_S

        return jnp.array([adot, ydot, sdot])

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        initial_state = jnp.ones((3,)) * 0.5

        init_state = jrandom.uniform(key,
                                     (3,),
                                     minval=0.5 - params.random_start_limit,
                                     maxval=0.5 + params.random_start_limit)
        init_state = init_state.at[2].set(0.5)

        state = jnp.where(self.defined_param_start and self.evaluating, initial_state, init_state)
        # TODO the above works but maybe theres a better way to do it? am sure takes some compute to do all the time

        env_state = EnvState(ays=state,
                             time=0)

        return self.get_obs(env_state), env_state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        ays = state_tp1.ays

        # Planetary Boundary Check
        inside_pb = jnp.logical_and(ays[0] < params.A_PB, ays[1] > params.Y_PB)
        PB_2 = jnp.array([params.A_PB, params.Y_PB, params.S_LIMIT])

        reward = jax.lax.select(inside_pb, jnp.linalg.norm(ays - PB_2),0.0)

        # Done flags
        green_fp_dist = jnp.all(jnp.abs(ays - params.green_fp) < params.final_radius)
        black_fp_dist = jnp.all(jnp.abs(ays - params.black_fp) < params.final_radius)

        arrived_at_final = jnp.logical_or(green_fp_dist, black_fp_dist)
        out_of_bounds = jnp.logical_or(ays[0] >= params.A_PB, ays[1] <= params.Y_PB)

        done = jnp.logical_or(arrived_at_final, out_of_bounds)
        fin_done = jnp.logical_or(done, state_tp1.time >= self.max_steps_in_ep)

        return reward, fin_done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()]

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return state.ays

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(ays=obs, time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "./animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        ays_history = np.asarray(trajectory_state.ays)
        times = np.asarray(trajectory_state.time)  # Extract time history
        A_vals = ays_history[:, 0]
        Y_vals = ays_history[:, 1]
        S_vals = ays_history[:, 2]

        fig = plt.figure(figsize=(10, 8))
        ax3d = fig.add_subplot(111, projection="3d")

        # Set axes limits based on min/max of the data so the plot doesn't jitter
        ax3d.set_xlim(0, 1)
        ax3d.set_ylim(0, 1)
        ax3d.set_zlim(0, 1)

        # Setup Axes
        azimuth, elevation = 170, 25
        ax3d.view_init(elevation, azimuth)
        ax3d.set_xlabel("\n\nexcess atmospheric carbon\nstock A [GtC]")
        ax3d.set_ylabel("\n\neconomic output Y\n[%1.0e USD/yr]" % 1e12)
        ax3d.set_zlabel("\n\nrenewable knowledge\nstock S [1e+09 Gj]")

        A_min, A_max = 0, params.A_PB
        Y_min, Y_max = params.Y_PB, 1
        S_min, S_max = params.S_LIMIT, 1
        plane_kwargs = {'color': 'grey', 'alpha': 0.15, 'shade': False, 'linewidth': 0}
        yy_A, zz_A = np.meshgrid([Y_min, Y_max], [S_min, S_max])
        xx_A = np.full_like(yy_A, params.A_PB)
        ax3d.plot_surface(xx_A, yy_A, zz_A, **plane_kwargs)
        xx_Y, zz_Y = np.meshgrid([A_min, A_max], [S_min, S_max])
        yy_Y = np.full_like(xx_Y, params.Y_PB)
        ax3d.plot_surface(xx_Y, yy_Y, zz_Y, **plane_kwargs)

        line, = ax3d.plot([], [], [], color='#377eb8', alpha=0.8, lw=3, label="Agent Trajectory")
        dot, = ax3d.plot([], [], [], color='red', marker='o', markersize=8, zorder=5)
        start_dot, = ax3d.plot([], [], [], color='green', marker='o', markersize=10, linestyle='None', label='Start', zorder=4)

        ax3d.legend()

        current_start_idx = 0

        def update(frame):
            nonlocal current_start_idx

            if times[frame] == 0:
                current_start_idx = frame

            start_dot.set_data([A_vals[current_start_idx]], [Y_vals[current_start_idx]])
            start_dot.set_3d_properties([S_vals[current_start_idx]])

            line.set_data(A_vals[current_start_idx: frame + 1], Y_vals[current_start_idx: frame + 1])
            line.set_3d_properties(S_vals[current_start_idx: frame + 1])

            dot.set_data([A_vals[frame]], [Y_vals[frame]])
            dot.set_3d_properties([S_vals[frame]])

            return line, dot, start_dot

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=ays_history.shape[0],
                                       interval=self.dt * 100,
                                       blit=False
                                       )

        save_path = f"{file_path}_{self.name}_trajectory.gif"
        anim.save(save_path)
        plt.close()

    @property
    def name(self) -> str:
        return "AYSIAM-v0"

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, (3,))


class AYSIAMCSCA(AYSIAMCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return action.squeeze() # TODO is it better to add clipping and scaling here with params than the hardcoded action space?

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(jnp.array((0.5, 0)),
                          jnp.array((1, 1)),
                          shape=(2,))
