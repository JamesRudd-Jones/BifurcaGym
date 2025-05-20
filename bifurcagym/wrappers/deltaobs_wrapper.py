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


class DeltaObsEnv(object):
    def __init__(self, wdeltaobs_env):
        """
        Returns delta obs rather than next observation.
        """
        self._wdeltaobs_env = wdeltaobs_env

    @property
    def wdeltaobs_env(self):
        return self._wdeltaobs_env

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[int, float, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        obs, new_env_state, rew, done, info = self._wdeltaobs_env.step(action, state, key)
        delta_obs = obs - self._wdeltaobs_env.get_obs(state)
        return delta_obs, new_env_state, rew, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: Union[int, float, chex.Array],
                        input_obs: chex.Array,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        obs, new_env_state, rew, done, info = self._wdeltaobs_env.generative_step(action,
                                                                                  input_obs,
                                                                                  key)
        delta_obs = obs - input_obs
        return delta_obs, new_env_state, rew, done, info

    def __getattr__(self, attr):
        if attr == "_wdeltaobs_env":
            raise AttributeError()
        return getattr(self._wdeltaobs_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wdeltaobs_env)