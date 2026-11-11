"""
Abstract base env for all further environments
"""

from functools import partial
from typing import Any, Dict, Tuple, Union
import chex
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jrandom
import abc


@struct.dataclass
class EnvState:
    time: int


def uniform_away_from_goal(key: chex.PRNGKey, low: chex.Array, high: chex.Array, dist_to_goal, exclusion: float):
    """
    Uniform sample over the box [low, high], resampled until dist_to_goal(sample) >= exclusion. Used for random starts.
    """
    def sample(carry):
        key, _ = carry
        key, _key = jrandom.split(key)
        return key, jrandom.uniform(_key, low.shape, minval=low, maxval=high, dtype=low.dtype)

    _, start = jax.lax.while_loop(lambda carry: dist_to_goal(carry[1]) < exclusion, sample, sample((key, low)))
    return start


class BaseEnvironment(abc.ABC):

    def __init__(self, **env_kwargs):
        # envs pop the kwargs they support before calling this, so anything left is a typo or unsupported option
        if env_kwargs:
            raise TypeError(f"{type(self).__name__} got unsupported env kwargs: {sorted(env_kwargs)}")

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[jnp.int_, jnp.float_, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        obs, delta_obs, state, reward, done, info = self.step_env(action, state, key)
        return obs, delta_obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: Union[jnp.int_, jnp.float_, chex.Array],
                        gen_obs: chex.Array,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        return self.step_env(action, self.get_state(gen_obs), key)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        obs, state = self.reset_env(key)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def apply_delta_obs(self, obs: chex.Array, delta_obs: chex.Array) -> chex.Array:
        return obs + delta_obs

    def step_env(self,
                 action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        raise NotImplementedError

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        raise NotImplementedError

    def reward_and_done_function(self,
                                 input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 key: chex.PRNGKey = None,
                                 )-> Tuple[chex.Array, chex.Array]:
        # TODO potential issues are any keyed random process added to the action as this may not travel to rewards correctly
        raise NotImplementedError

    def get_obs(self, state, key: chex.PRNGKey = None) -> chex.Array:
        raise NotImplementedError

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        raise NotImplementedError

    def discount(self, done: chex.Array) -> chex.Array:
        return jax.lax.select(done, 0.0, 1.0)

    def render_traj(self, state: EnvState, file_path: str):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

    def action_space(self):
        raise NotImplementedError

    def observation_space(self):
        raise NotImplementedError

    def reward_space(self):
        raise NotImplementedError
