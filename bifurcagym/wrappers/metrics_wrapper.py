import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
import chex
from typing import Any, Dict, Tuple, Union
from bifurcagym.envs import EnvState
from flax import struct


@struct.dataclass
class MetricsEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class MetricsWrapper(object):
    """
        Logs env metrics for later used, greatly inspired by the following from PureJAXRL
        https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py
    """

    def __init__(self, wrapped_metrics_env):
        self._wrapped_metrics_env = wrapped_metrics_env

    @property
    def wrapped_metrics_env(self):
        return self._wrapped_metrics_env

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[jnp.int_, jnp.float_, chex.Array],
             state: MetricsEnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, MetricsEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        obs, delta_obs, env_state, reward, done, info = self._wrapped_metrics_env.step(action, state.env_state, key)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = MetricsEnvState(env_state=env_state,
                                episode_returns=new_episode_return * (1 - done),
                                episode_lengths=new_episode_length * (1 - done),
                                returned_episode_returns=state.returned_episode_returns * (1 - done)
                                + new_episode_return * done,
                                returned_episode_lengths=state.returned_episode_lengths * (1 - done)
                                + new_episode_length * done,
                                timestep=state.timestep + 1,
                                )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, delta_obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, MetricsEnvState]:
        obs, env_state = self._wrapped_metrics_env.reset(key)
        state = MetricsEnvState(env_state,
                                episode_returns=0.0,
                                episode_lengths=0,
                                returned_episode_returns=0.0,
                                returned_episode_lengths=0,
                                timestep=0)
        return obs, state

    def __getattr__(self, attr):
        if attr == "_wrapped_metrics_env":
            raise AttributeError()
        return getattr(self._wrapped_metrics_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_metrics_env)