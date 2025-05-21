from gymnax.environments import spaces
from copy import deepcopy
import jax.numpy as jnp
import jax.random as jrandom
import jax
from functools import partial
import chex
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, overload
from bifurcagym.envs import base_env
from bifurcagym.envs.base_env import EnvState


class PeriodicEnv(object):
    """
    Shifts obs when there are periodic boundaries.
    """
    def __init__(self, wperiodic_env):

        self._wperiodic_env = wperiodic_env

    # @property
    # def wperiodic_env(self):
    #     return self._wperiodic_env

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[int, float, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        obs, delta_obs, new_env_state, rew, done, info = self._wperiodic_env.step(action, state, key)

        shifted_output_og = obs - self._wperiodic_env.observation_space().low
        obs_range = self._wperiodic_env.observation_space().high - self._wperiodic_env.observation_space().low
        shifted_output = jnp.remainder(shifted_output_og, obs_range)
        modded_output = shifted_output_og + (self._wperiodic_env.periodic_dim * shifted_output) - (
                    self._wperiodic_env.periodic_dim * shifted_output_og)
        wrapped_output = modded_output + self._wperiodic_env.observation_space().low

        # TODO this adds in some error accumulation, can we avoid this?

        return wrapped_output, delta_obs, new_env_state, rew, done, info

    def __getattr__(self, attr):
        if attr == "_wperiodic_env":
            raise AttributeError()
        return getattr(self._wperiodic_env, attr)

    # def __str__(self):
    #     return "{}({})".format(type(self).__name__, self.wperiodic_env)