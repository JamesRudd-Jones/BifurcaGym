import jax
import jax.numpy as jnp
from functools import partial
import chex


class PeriodicEnv(object):
    """
    Shifts observations when there are periodic boundaries. Useful when predicting delta_obs as without it the generated
    observations do not conform to the bounds of the simulation.
    """
    def __init__(self, wrapped_periodic_env):

        self._wrapped_periodic_env = wrapped_periodic_env

    @property
    def wrapped_periodic_env(self):
        return self._wrapped_periodic_env

    @partial(jax.jit, static_argnums=(0,))
    def apply_delta_obs(self, obs: chex.Array, delta_obs: chex.Array) -> chex.Array:
        shifted_output_og = (self._wrapped_periodic_env.apply_delta_obs(obs, delta_obs) -
                             self._wrapped_periodic_env.observation_space().low)
        obs_range = (self._wrapped_periodic_env.observation_space().high -
                     self._wrapped_periodic_env.observation_space().low)
        shifted_output = jnp.remainder(shifted_output_og, obs_range)
        modded_output = shifted_output_og + (self._wrapped_periodic_env.periodic_dim * shifted_output) - (
                    self._wrapped_periodic_env.periodic_dim * shifted_output_og)
        wrapped_output = modded_output + self._wrapped_periodic_env.observation_space().low

        # TODO this adds in some error accumulation, can we avoid this?

        return wrapped_output

    def __getattr__(self, attr):
        if attr == "_wrapped_periodic_env":
            raise AttributeError()
        return getattr(self._wrapped_periodic_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_periodic_env)