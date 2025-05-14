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


TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")


@struct.dataclass
class EnvState:
    time: int


class BaseEnvironment(Generic[TEnvState, TEnvParams]):  # object):
    """Jittable abstract base class for all gymnax Environments."""

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    # TODO would be cool to have all params here
    # TODO then figure out a way to replace params without having to track env_params everywhere

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             key: chex.PRNGKey,
             state: TEnvState,
             action: Union[int, float, chex.Array]
             ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        key, key_reset = jrandom.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action)
        obs_re, state_re = self.reset_env(key_reset)
        # Auto-reset environment based on termination
        state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        key: chex.PRNGKey,
                        gen_obs: chex.Array,
                        action: Union[int, float, chex.Array]
                        ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs generative step transition in the environment for some Model-Based RL approaches."""
        key, key_reset = jrandom.split(key)
        obs_st, state_st, reward, done, info = self.generative_step_env(key, gen_obs, action)
        obs_re, state_re = self.reset_env(key_reset)
        # Auto-reset environment based on termination
        state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, TEnvState]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(key)
        return obs, state

    def step_env(self,
                 key: chex.PRNGKey,
                 state: TEnvState,
                 action: Union[int, float, chex.Array]
                ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def generative_step_env(self,
                            key: chex.PRNGKey,
                            gen_obs: chex.Array,
                            action: Union[int, float, chex.Array]
                            ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reward_func(self,
                    key: chex.PRNGKey,
                    x_t: chex.Array,
                    x_tp1: chex.Array
                    )-> chex.Array:  # TODO is it an array idk?
        """Environment-specific reward function."""
        raise NotImplementedError

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, TEnvState]:
        """Environment-specific reset."""
        raise NotImplementedError

    def render_env(self, key: chex.PRNGKey, state: TEnvState):
        """Environment-specific reset."""
        raise NotImplementedError

    @overload
    def get_obs(self,
                state: TEnvState,
                ) -> chex.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    @overload
    def get_obs(self,
                state: TEnvState,
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

    def is_done(self, state: TEnvState) -> jnp.ndarray:
        """Check whether state transition is done."""
        raise NotImplementedError

    def discount(self, state: TEnvState) -> jnp.ndarray:
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