"""
Abstract base env for all further environments
"""

from functools import partial
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, overload
import chex
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jrandom
import abc


@struct.dataclass
class EnvState:
    time: int


class BaseEnvironment(abc.ABC):  # object):

    def __init(self, **env_kwargs):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[int, float, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        obs, state, reward, done, info = self.step_env(action, state, key)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: Union[int, float, chex.Array],
                        gen_obs: chex.Array,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        obs, state, reward, done, info = self.generative_step_env(action, gen_obs, key)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> tuple[chex.Array, EnvState]:
        obs, state = self.reset_env(key)
        return obs, state

    def step_env(self,
                 action: Union[int, float, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        raise NotImplementedError

    def generative_step_env(self,
                            action: Union[int, float, chex.Array],
                            gen_obs: chex.Array,
                            key: chex.PRNGKey,
                            ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        raise NotImplementedError

    def reset_env(self, key: chex.PRNGKey) -> tuple[chex.Array, EnvState]:
        raise NotImplementedError

    def reward_func(self,
                    input_action_t: Union[int, float, chex.Array],
                    state_t: EnvState,
                    state_tp1: EnvState,
                    key: chex.PRNGKey,
                    )-> chex.Array:
        # TODO this is needed for discrete actions as we may pass actions from outside the env
        # TODO potential issues are any keyed random process added to the action as this may not travel to rewards correctly

        raise NotImplementedError

    def render(self, state: EnvState):
        raise NotImplementedError

    @overload
    def get_obs(self,
                state: EnvState,
                ) -> chex.Array:
        raise NotImplementedError

    @overload
    def get_obs(self,
                state: EnvState,
                key: chex.PRNGKey
                ) -> chex.Array:
        raise NotImplementedError

    def get_obs(self,
                state,
                key=None
                ) -> chex.Array:
        raise NotImplementedError

    def is_done(self, state: EnvState) -> chex.Array:
        raise NotImplementedError

    def discount(self, state: EnvState) -> chex.Array:
        return jax.lax.select(self.is_done(state), 0.0, 1.0)

    @property
    def name(self) -> str:
        return type(self).__name__

    def action_space(self):
        raise NotImplementedError

    def observation_space(self):
        raise NotImplementedError
