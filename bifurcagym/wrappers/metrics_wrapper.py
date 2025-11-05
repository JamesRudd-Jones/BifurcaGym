import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
import chex
from typing import Any, Dict, Tuple, Union
from bifurcagym.envs import EnvState
from flax import struct
import dataclasses


@struct.dataclass
class MetricsEnvState:
    env_state: EnvState
    episode_lengths: int
    episode_returns: float
    returned_episode_lengths: int
    returned_episode_returns: float
    timestep: int


class MetricsWrapper(object):
    """
        Logs env metrics for later use, greatly inspired by the following from PureJAXRL
        https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py
    """

    def __init__(self, wrapped_metrics_env):
        self._wrapped_metrics_env = wrapped_metrics_env

        _, env_state = self._wrapped_metrics_env.reset(jrandom.key(0))

        self._base_env_cls = env_state
        self._base_fields = {f.name for f in dataclasses.fields(self._base_env_cls)}

    @property
    def wrapped_metrics_env(self):
        return self._wrapped_metrics_env

    def _to_base_env_state(self, combined_state: MetricsEnvState) -> EnvState:
        base_kwargs = {name: getattr(combined_state, name) for name in self._base_fields}
        return self._base_env_cls(**base_kwargs)

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[jnp.int_, jnp.float_, chex.Array],
             state: MetricsEnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, MetricsEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        # base_env_state = self._to_base_env_state(state)
        # obs, delta_obs, new_base_env_state, reward, done, info = self._wrapped_metrics_env.step(action, base_env_state, key)
        # base_dict = dataclasses.asdict(new_base_env_state)
        base_env_state = self._to_base_env_state(state)
        obs, delta_obs, nenv_state, reward, done, info = self._wrapped_metrics_env.step(action, state.env_state, key)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        # state = MetricsEnvState(**base_dict,
        #                         episode_lengths=new_episode_length * (1 - done),
        #                         episode_returns=new_episode_return * (1 - done),
        #                         returned_episode_lengths=state.returned_episode_lengths * (1 - done)
        #                                                  + new_episode_length * done,
        #                         returned_episode_returns=state.returned_episode_returns * (1 - done)
        #                         + new_episode_return * done,
        #                         timestep=state.timestep + 1,
        #                         )
        state = MetricsEnvState(nenv_state,
                                episode_lengths=new_episode_length * (1 - done),
                                episode_returns=new_episode_return * (1 - done),
                                returned_episode_lengths=state.returned_episode_lengths * (1 - done)
                                                         + new_episode_length * done,
                                returned_episode_returns=state.returned_episode_returns * (1 - done)
                                                         + new_episode_return * done,
                                timestep=state.timestep + 1,
                                )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, delta_obs, state, reward, done, info
        # TODO how best can I do this metrics wrapper to avoid the env_state.env_state thing that only exists when I use the metrics wrapper?
        # TODO it is a challenging problem

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, MetricsEnvState]:
        obs, base_env_state = self._wrapped_metrics_env.reset(key)
        # base_dict = dataclasses.asdict(base_env_state)
        # state = MetricsEnvState(**base_dict,
        #                         episode_lengths=0,
        #                         episode_returns=0.0,
        #                         returned_episode_lengths=0,
        #                         returned_episode_returns=0.0,
        #                         timestep=0)
        state = MetricsEnvState(base_env_state,
                                episode_lengths=0,
                                episode_returns=0.0,
                                returned_episode_lengths=0,
                                returned_episode_returns=0.0,
                                timestep=0)
        return obs, state

    def __getattr__(self, attr):
        if attr == "_wrapped_metrics_env":
            raise AttributeError()
        return getattr(self._wrapped_metrics_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_metrics_env)