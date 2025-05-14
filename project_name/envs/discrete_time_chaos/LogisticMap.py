import numpy as np
from os import path
import jax.numpy as jnp
import jax.random as jrandom
from gymnax.environments import environment
from gymnax.environments import spaces
from flax import struct
from typing import Any, Dict, Optional, Tuple, Union
import chex
import jax


@struct.dataclass
class EnvState(environment.EnvState):
    x: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    # init_r: float = 2.5
    # init_r: float = 3.1
    init_r: float = 3.8
    # init_r: float = 4.0
    # fixed_point: float = 0.6  # for r = 2.5; period 1
    # fixed_point: float = 0.55801  # for r = 3.1; period 2
    # fixed_point: float = 0.76457  # for r = 3.1; period 2
    # fixed_point: float = 0.67742  # for r = 3.1; period 1
    fixed_point: float = 0.737  # for r = 3.8; chaotic
    # fixed_point: float = 0.xxx  # for r = 4.0; chaotic

    reward_ball: float = 0.001  # 0.0025  # 0.005
    # reward_ball: float = 0.01  # 0.0025  # 0.005
    max_control: float = 0.1
    horizon: int = 200
    start_point: float = 0.1
    action_array: jnp.ndarray = jnp.array(((0,), (0.1,), (-0.1,)))
    ref_vector: jnp.ndarray = jnp.linspace(0, 1, 100 + 1)
    discrete_action: bool = True
    A_MAX = 0.1

class GymnaxLogisticMap(environment.Environment[EnvState, EnvParams]):

    def __init__(self):
        super().__init__()
        self.obs_dim = 1

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        # TODO some calc to automatically figure out n period fixed point for a specific value of r

        return EnvParams()

    def step_env(self,
                 key: chex.PRNGKey,
                 state: EnvState,
                 action_idx: Union[int, float, chex.Array],
                 params: EnvParams) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # u = jax.lax.cond(params.discrete_action, self.discrete_action, self.continuous_action, action_idx, params)
        u = params.action_array[action_idx]
        # u = params.action_array[0]
        # u = action_idx

        # u = jnp.clip(action, -params.max_control, params.max_control)

        new_x = (u + params.init_r) * state.x * (1 - state.x)

        # reward = jax.lax.select(jnp.abs(new_x - params.fixed_point) < params.reward_ball, jnp.ones(1, ), jnp.zeros(1, ))
        # reward = -jnp.square(new_x - params.fixed_point)
        reward = -jnp.abs(new_x - params.fixed_point) ** 2  # TODO can set more specific norm distances

        fake_done = jax.lax.select(jnp.abs(new_x - params.fixed_point) < params.reward_ball, jnp.array((True,)), jnp.array((False,)))
        # reward = jax.lax.select(fake_done, jnp.zeros(1, ), -jnp.ones(1, ))

        done = jnp.array(False)
        # done = jax.lax.select(jnp.squeeze(w == nw), True, False)
        # done = jax.lax.select(jnp.squeeze(state.x == new_x), True, False)
        # old done is above for continuous env

        state = EnvState(x=new_x, time=state.time+1)

        delta_s = jnp.zeros(1,)  # TODO add in some delta s value

        return (jax.lax.stop_gradient(self.get_obs(state, params)),
                jax.lax.stop_gradient(state),
                reward,
                done,
                {"delta_obs": delta_s,
                 "fake_done": jnp.squeeze(fake_done)})

    def discrete_action(self, action_idx, params):
        return params.action_array[action_idx]

    def continuous_action(self, action_idx, params):
        return action_idx

    def generative_step_env(self, key, obs, action, params):
        state = EnvState(x=obs[0], time=0)
        return self.step_env(key, state, action, params)

    def reward_function(self, x, next_obs, params: EnvParams):
        """
        As per the paper titled: Optimal chaos control through reinforcement learning
        """
        reward = jax.lax.select(x == next_obs, jnp.zeros(1,), -jnp.ones(1,))
        return reward

    def reset_env(self, key: chex.PRNGKey, params: EnvParams, random: bool = True) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=jnp.array((params.start_point,)), time=0)
        random_state = EnvState(x=jrandom.uniform(_key,
                                                  shape=(1,),
                                                  minval=params.start_point - 0.05,
                                                  maxval=params.start_point + 0.05),
                                time=0)
        random_state = EnvState(x=jrandom.uniform(_key, shape=(1,), minval=0.0, maxval=1.0), time=0)

        state = jax.tree_map(lambda x, y: jax.lax.select(random, x, y), random_state, same_state)

        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return self.projection(state.x, params)

    def projection(self, s, params):
        # TODO only for 1d atm
        s = jnp.repeat(s, params.ref_vector.shape[0])
        inter = jnp.abs(params.ref_vector - s)
        return jnp.argmin(inter)

    @property
    def name(self) -> str:
        """Environment name."""
        return "LogisticMap-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-params.max_control, params.max_control, shape=(1,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([1.0])
        return spaces.Box(-high, high, (1,), dtype=jnp.float32)

    # TODO add in state space