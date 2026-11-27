"""
A minimal system driven by a chaotic driver:

    x' = a x + w + u,    w' = mu min(w, 1 - w)    (a tent map driver)

x on its own contracts (|a| < 1), all the chaos is in w, which the agent cannot change. The task is to hold x at 0,
which means cancelling the driver: with w observed, u = -(a x + w) does it in one step. With `observe_driver=False`
the agent sees only x and has to infer the driver from x's history (w_t = x_{t+1} - a x_t - u_t, then w evolves by the
known tent map), so it needs memory and the chaos amplifies any error in that inference.
"""

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
    w: jnp.ndarray


@struct.dataclass
class EnvParams:
    a: float = 0.5  # contraction of x
    mu: float = 1.99  # the tent driver, 2 exactly collapses every floating point orbit onto 0

    max_control: float = 2.0
    maximum_max_control: float = struct.field(False, default=2.0)


class SkewProductCSCA(base_env.BaseEnvironment):

    def __init__(self, observe_driver: bool = True, **env_kwargs):
        super().__init__(**env_kwargs)

        self.observe_driver: bool = observe_driver

        self.horizon: int = 200
        self.max_steps_in_ep: int = 1000

        self.action_array: chex.Array = jnp.array((0.0, 0.5, -0.5, 1.0, -1.0))

        self.requires_float64: bool = True

        self.reward_scale: float = 1.0  # mean |x| uncontrolled, E[w] / (1 - a)

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

        new_x = params.a * state.x + state.w + action
        new_w = params.mu * jnp.minimum(state.w, 1 - state.w)
        new_state = EnvState(x=new_x, w=new_w, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (self.get_obs(new_state),
                self.get_obs(new_state) - self.get_obs(state),
                new_state,
                reward,
                done,
                {})

    def generative_step(self,
                        action: chex.Numeric,
                        gen_obs: chex.Array,
                        params: EnvParams,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        if not self.observe_driver:
            raise ValueError(f"No Generative Step for {self.name} with the driver hidden, x alone does not give w.")

        return super().generative_step(action, gen_obs, params, key)

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        x_key, w_key = jrandom.split(key)
        state = EnvState(x=jrandom.uniform(x_key, (), minval=0.0, maxval=2.0),
                         w=jrandom.uniform(w_key, (), minval=0.0, maxval=1.0),
                         time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        reward = jnp.exp(-jnp.abs(state_tp1.x) / self.reward_scale)

        done = jnp.asarray(state_tp1.time >= self.max_steps_in_ep)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control).squeeze()

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        if self.observe_driver:
            return jnp.array((state.x, state.w))

        return jnp.array((state.x,))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(x=obs[0], w=obs[1], time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        xs = np.asarray(trajectory_state.x).flatten()
        ws = np.asarray(trajectory_state.w).flatten()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"{self.name}, driver {'observed' if self.observe_driver else 'hidden'}")
        ax.set_xlim(0, len(xs))
        ax.set_ylim(min(-0.5, xs.min() - 0.1), max(2.0, xs.max() + 0.1))
        ax.set_xlabel("Time Step (n)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0.0, color='k', linestyle='--', lw=1.0, zorder=1)

        x_line, = ax.plot([], [], color='blue', lw=1.0, label="x", zorder=3)
        w_line, = ax.plot([], [], color='grey', lw=0.8, alpha=0.7, label="driver w", zorder=2)
        ax.legend()

        def update(frame):
            x_line.set_data(np.arange(frame + 1), xs[:frame + 1])
            w_line.set_data(np.arange(frame + 1), ws[:frame + 1])
            return x_line, w_line

        anim = animation.FuncAnimation(fig, update, frames=len(xs), interval=50, blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "SkewProduct-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(1,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        # Uncontrolled x stays in [0.19, 1.6], controlled it can be driven negative, so the box allows both
        if self.observe_driver:
            return spaces.Box(jnp.array((-2.0, 0.0)), jnp.array((2.0, 1.0)), (2,), dtype=jnp.float64)

        return spaces.Box(-2.0, 2.0, (1,), dtype=jnp.float64)


class SkewProductCSDA(SkewProductCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
