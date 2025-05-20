from gymnax.environments import spaces
from copy import deepcopy
import jax.numpy as jnp
import jax.random as jrandom
import jax
from functools import partial
import chex
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, overload
from project_name.envs import base_env
from project_name.envs.base_env import EnvState


class AutoResetWrapper(object):
    def __init__(self, wautoreset_env):
        """
        Automatically resets envs that are done, greatly inspired by the following from Gymnax
        https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py

        Also taking insight from Brax:
        https://github.com/google/brax/blob/main/brax/envs/wrappers/training.py
        """

        self._wautoreset_env = wautoreset_env

    @property
    def wautoreset_env(self):
        return self._wautoreset_env

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[int, float, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        obs_st, state_st, reward, done, info = self._wautoreset_env.step(action, state, key)
        key, key_reset = jrandom.split(key)
        obs_re, state_re = self._wautoreset_env.reset(key_reset)
        # Auto-reset environment based on termination
        state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: Union[int, float, chex.Array],
                        gen_obs: chex.Array,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        obs_st, state_st, reward, done, info = self._wautoreset_env.generative_step(action, gen_obs, key)
        key, key_reset = jrandom.split(key)
        obs_re, state_re = self._wautoreset_env.reset(key_reset)
        # Auto-reset environment based on termination
        state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def __getattr__(self, attr):
        if attr == "_wautoreset_env":
            raise AttributeError()
        return getattr(self._wautoreset_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wautoreset_env)