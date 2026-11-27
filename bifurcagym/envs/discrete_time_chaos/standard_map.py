"""
Chirikov's standard map on the torus [-pi, pi)^2
"""

from typing import Any, Dict, Tuple

import chex
from flax import struct
import jax.numpy as jnp
import jax.random as jrandom

from bifurcagym import spaces
from bifurcagym.envs import base_env
from bifurcagym.envs.discrete_time_chaos.cat_map import torus_distance


@struct.dataclass
class EnvState(base_env.EnvState):
    theta: jnp.ndarray
    p: jnp.ndarray


@struct.dataclass
class EnvParams:
    K: float = 1.0

    max_control: float = 0.05
    maximum_max_control: float = struct.field(False, default=0.05)


def wrap(x: chex.Array) -> chex.Array:
    """Into [-pi, pi)"""
    return jnp.mod(x + jnp.pi, 2 * jnp.pi) - jnp.pi


class StandardMapCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.horizon: int = 200
        self.max_steps_in_ep: int = 1000

        self.action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))

        self.requires_float64: bool = True

        self.periodic_dim: chex.Array = jnp.ones(2)  # theta and p both live on [-pi, pi)

        self.fixed_point: chex.Array = jnp.zeros(2)  # (theta, p), a saddle for K > 0
        self.start_box: float = 0.5
        self.reward_scale: float = 1.64

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

        new_p = wrap(state.p + params.K * jnp.sin(state.theta) + action)
        new_theta = wrap(state.theta + new_p)
        new_state = EnvState(theta=new_theta, p=new_p, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (self.get_obs(new_state),
                self.get_obs(new_state) - self.get_obs(state),
                new_state,
                reward,
                done,
                {})

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        start = jrandom.uniform(key, (2,), minval=-self.start_box, maxval=self.start_box)
        state = EnvState(theta=start[0], p=start[1], time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        distance = torus_distance(jnp.array((state_tp1.theta, state_tp1.p)), self.fixed_point, 2 * jnp.pi)
        reward = jnp.exp(-distance / self.reward_scale)

        done = jnp.asarray(state_tp1.time >= self.max_steps_in_ep)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control).squeeze()

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.theta, state.p))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(theta=obs[0], p=obs[1], time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        thetas = np.asarray(trajectory_state.theta)
        ps = np.asarray(trajectory_state.p)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(self.name)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel("theta")
        ax.set_ylabel("p")
        ax.set_aspect("equal")
        ax.plot(0.0, 0.0, 'kx', markersize=10, label="Saddle", zorder=3)

        points, = ax.plot([], [], color='blue', linestyle='', marker='.', markersize=3, alpha=0.6, zorder=1)
        dot, = ax.plot([], [], color='red', marker='o', markersize=6, zorder=2)
        ax.legend()

        def update(frame):
            points.set_data(thetas[:frame + 1], ps[:frame + 1])
            dot.set_data([thetas[frame]], [ps[frame]])
            return points, dot

        anim = animation.FuncAnimation(fig, update, frames=thetas.shape[0], interval=100, blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "StandardMap-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(1,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-jnp.pi, jnp.pi, (2,), dtype=jnp.float64)


class StandardMapCSDA(StandardMapCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
