import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
import chex
from typing import Any, Dict, Tuple, Union
from bifurcagym.envs import EnvState


class VMapWrapper(object):
    def __init__(self, wrapped_vmappable_env):

        self._wrapped_vmappable_env = wrapped_vmappable_env

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[jnp.int_, jnp.float_, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        return jax.vmap(self._wrapped_vmappable_env.step)(action, state, key)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        return jax.vmap(self._wrapped_vmappable_env.reset)(key)

    def __getattr__(self, attr):
        if attr == "_wrapped_vmappable_env":
            raise AttributeError()
        return getattr(self._wrapped_vmappable_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_vmappable_env)