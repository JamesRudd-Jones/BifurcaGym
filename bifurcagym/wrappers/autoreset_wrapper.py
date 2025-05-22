import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
import chex
from typing import Any, Dict, Tuple, Union
from bifurcagym.envs import EnvState


class AutoResetWrapper(object):
    def __init__(self, wrapped_autoreset_env):
        """
        Automatically resets envs that are done, greatly inspired by the following from Gymnax
        https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py

        And also from Brax:
        https://github.com/google/brax/blob/main/brax/envs/wrappers/training.py
        """

        self._wrapped_autoreset_env = wrapped_autoreset_env

    @property
    def wrapped_autoreset_env(self):
        return self._wrapped_autoreset_env

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[jnp.int_, jnp.float_, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        obs_st, delta_obs_st, state_st, reward, done, info = self._wrapped_autoreset_env.step(action, state, key)
        key, key_reset = jrandom.split(key)
        obs_re, state_re = self._wrapped_autoreset_env.reset(key_reset)
        # Auto-reset environment based on termination
        state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        delta_obs = jax.lax.select(done, jnp.zeros_like(delta_obs_st), delta_obs_st)
        # TODO unsure if this best approach will think about it more concretely
        return obs, delta_obs, state, reward, done, info

    def __getattr__(self, attr):
        if attr == "_wrapped_autoreset_env":
            raise AttributeError()
        return getattr(self._wrapped_autoreset_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_autoreset_env)