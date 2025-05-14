import jax
import chex
from functools import partial
from typing import Union, Dict, Any, Optional


class GenerativeEnv(object):
    def __init__(self, wrapper_env, env_params):
        self._wrapper_env = wrapper_env
        self.env_params = env_params

    @property
    def wrapper_env(self):
        return self._wrapper_env

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self, key: chex.PRNGKey, orig_obs, action: Union[int, float, chex.Array],
                        params: Optional[TEnvParams] = None) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self._wrapper_env.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self._wrapper_env.generative_step_env(key, orig_obs, action, params)
        obs_re, state_re = self._wrapper_env.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def __getattr__(self, attr):
        if attr == "_wrapper_env":
            raise AttributeError()
        return getattr(self._wrapper_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapper_env)