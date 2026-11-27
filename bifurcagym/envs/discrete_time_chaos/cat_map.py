from typing import Any, Dict, Tuple

import chex
from flax import struct
import jax.numpy as jnp
import jax.random as jrandom

from bifurcagym import spaces
from bifurcagym.envs import base_env


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray


@struct.dataclass
class EnvParams:
    max_control: float = 0.02
    maximum_max_control: float = struct.field(False, default=0.02)


def torus_distance(xy: chex.Array, target: chex.Array, period: float) -> chex.Array:
    """Euclidean distance on a torus of side `period`"""
    delta = jnp.abs(xy - target)

    return jnp.linalg.norm(jnp.minimum(delta, period - delta))


class CatMapCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.horizon: int = 200
        self.max_steps_in_ep: int = 1000

        self.action_array: chex.Array = jnp.array(((0.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                                                   (1.0, 0.0), (1.0, 1.0), (1.0, -1.0),
                                                   (-1.0, 0.0), (-1.0, 1.0), (-1.0, -1.0)))

        self.requires_float64: bool = True

        self.periodic_dim: chex.Array = jnp.ones(2)

        self.fixed_point: chex.Array = jnp.zeros(2)
        self.reward_scale: float = 0.38  # mean torus distance of the uncontrolled (uniformly spread) orbit from (0, 0)

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

        new_x = jnp.mod(2 * state.x + state.y + action[0], 1.0)
        new_y = jnp.mod(state.x + state.y + action[1], 1.0)
        new_state = EnvState(x=new_x, y=new_y, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (self.get_obs(new_state),
                self.get_obs(new_state) - self.get_obs(state),
                new_state,
                reward,
                done,
                {})

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        xy = jrandom.uniform(key, (2,), minval=0.0, maxval=1.0)
        state = EnvState(x=xy[0], y=xy[1], time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        distance = torus_distance(jnp.array((state_tp1.x, state_tp1.y)), self.fixed_point, 1.0)
        reward = jnp.exp(-distance / self.reward_scale)

        done = jnp.asarray(state_tp1.time >= self.max_steps_in_ep)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x, state.y))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        xs = np.asarray(trajectory_state.x)
        ys = np.asarray(trajectory_state.y)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(self.name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.plot(0.0, 0.0, 'kx', markersize=10, label="Fixed Point", zorder=3)

        points, = ax.plot([], [], color='blue', linestyle='', marker='.', markersize=3, alpha=0.6, zorder=1)
        dot, = ax.plot([], [], color='red', marker='o', markersize=6, zorder=2)
        ax.legend()

        def update(frame):
            points.set_data(xs[:frame + 1], ys[:frame + 1])
            dot.set_data([xs[frame]], [ys[frame]])
            return points, dot

        anim = animation.FuncAnimation(fig, update, frames=xs.shape[0], interval=100, blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "CatMap-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(2,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (2,), dtype=jnp.float64)


class CatMapCSDA(CatMapCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
