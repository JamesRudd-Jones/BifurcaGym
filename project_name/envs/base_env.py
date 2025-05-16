"""
Abstract base env for all further Environments
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
             ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        obs, state, reward, done, info = self.step_env(action, state, key)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: Union[int, float, chex.Array],
                        gen_obs: chex.Array,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
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
                ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def generative_step_env(self,
                            action: Union[int, float, chex.Array],
                            gen_obs: chex.Array,
                            key: chex.PRNGKey,
                            ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(self, key: chex.PRNGKey) -> tuple[chex.Array, EnvState]:
        """Environment-specific reset."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def reward_func(self,
                    x_t: chex.Array,
                    x_tp1: chex.Array,
                    key: chex.PRNGKey,
                    )-> chex.Array:  # TODO is it an array idk?
        """Environment-specific reward function."""
        raise NotImplementedError

    def render(self, state: EnvState):
        """Environment-specific reset."""
        raise NotImplementedError

    @overload
    def get_obs(self,
                state: EnvState,
                ) -> chex.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    @overload
    def get_obs(self,
                state: EnvState,
                key: chex.PRNGKey
                ) -> chex.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    def get_obs(self,
                state,
                key=None
                ) -> chex.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def is_done(self, state: EnvState) -> jnp.ndarray:
        """Check whether state transition is done."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def discount(self, state: EnvState) -> jnp.ndarray:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_done(state), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    def action_space(self):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self):
        """Observation space of the environment."""
        raise NotImplementedError

    # def state_space(self):
    #     """State space of the environment."""
    #     raise NotImplementedError
    # TODO do we need to define the above idk