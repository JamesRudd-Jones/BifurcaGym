from gymnax.environments import spaces
from copy import deepcopy
import jax.numpy as jnp
import jax.random as jrandom
import jax
from functools import partial
import chex
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, overload
from project_name.envs import base_env


TEnvState = TypeVar("TEnvState", bound="EnvState")


class NormalisedEnv(base_env.BaseEnvironment):
    def __init__(self, wrapped_env):
        """
        Normalises obs to be between -1 and 1
        """
        self._wrapped_env = wrapped_env
        self.unnorm_action_space = self._wrapped_env.action_space()
        self.unnorm_observation_space = self._wrapped_env.observation_space()
        self.unnorm_obs_space_size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        self.unnorm_action_space_size = self.unnorm_action_space.high - self.unnorm_action_space.low

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def action_space(self) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-1, 1, shape=self.unnorm_action_space.shape)

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(low=-jnp.ones_like(self.unnorm_observation_space.low,),
                          high=jnp.ones_like(self.unnorm_observation_space.high,),
                          shape=self.unnorm_observation_space.shape)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, TEnvState]:
        unnorm_obs, env_state = self._wrapped_env.reset(key)
        return self.normalise_obs(unnorm_obs), env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[int, float, chex.Array],
             state: TEnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, new_env_state, rew, done, info = self._wrapped_env.step(key, state, unnorm_action)

        unnorm_delta_obs = info["delta_obs"]
        norm_delta_obs = unnorm_delta_obs / self.unnorm_obs_space_size * 2
        info["delta_obs"] = norm_delta_obs

        return self.normalise_obs(unnorm_obs), new_env_state, rew, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: Union[int, float, chex.Array],
                        norm_obs: chex.Array,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        unnorm_init_obs = self.unnormalise_obs(norm_obs)
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, new_env_state, rew, done, info = self._wrapped_env.generative_step(key,
                                                                                       unnorm_init_obs,
                                                                                       unnorm_action)

        unnorm_delta_obs = info["delta_obs"]
        norm_delta_obs = unnorm_delta_obs / self.unnorm_obs_space_size * 2
        info["delta_obs"] = norm_delta_obs

        return self.normalise_obs(unnorm_obs), new_env_state, rew, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reward_func(self,
                    x_t: chex.Array,
                    x_tp1: chex.Array,
                    key: chex.PRNGKey,
                    ) -> chex.Array:  # TODO is it an array idk?
        norm_obs = x_t[..., :self._wrapped_env.obs_dim]
        action = x_t[..., self._wrapped_env.obs_dim:]
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs = self.unnormalise_obs(norm_obs)
        unnorm_x = jnp.concatenate([unnorm_obs, unnorm_action], axis=-1)
        unnorm_y = self.unnormalise_obs(x_tp1)
        rewards = self._wrapped_env.reward_function(unnorm_x, unnorm_y)

        return rewards

    # TODO might need a render normalise as well

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_env)

    def normalise_obs(self,  obs: chex.Array) -> chex.Array:
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1

        return norm_obs

    def unnormalise_obs(self, obs: chex.Array) -> chex.Array:
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        obs01 = (obs + 1) / 2
        obs_ranged = obs01 * size
        unnorm_obs = obs_ranged + low

        return unnorm_obs

    def unnormalise_action(self, action: Union[int, float, chex.Array]) -> Union[int, float, chex.Array]:
        low = self.unnorm_action_space.low
        size = self.unnorm_action_space_size
        act01 = (action + 1) / 2
        act_ranged = act01 * size
        unnorm_act = act_ranged + low

        return unnorm_act

    def normalise_action(self, action: Union[int, float, chex.Array]) -> Union[int, float, chex.Array]:
        low = self.unnorm_action_space.low
        size = self.unnorm_action_space_size
        pos_action = action - low
        norm_action = (pos_action / size * 2) - 1

        return norm_action


def make_normalised_plot_fn(norm_env, env_params, plot_fn):
    obs_dim = norm_env.observation_space().low.size
    wrapped_env = norm_env.wrapped_env
    # Set domain
    low = np.concatenate([wrapped_env.observation_space(env_params).low,
                          np.expand_dims(np.array(wrapped_env.action_space(env_params).low), axis=0)])
    high = np.concatenate([wrapped_env.observation_space(env_params).high,
                           np.expand_dims(np.array(wrapped_env.action_space(env_params).high), axis=0)])
    unnorm_domain = [elt for elt in zip(low, high)]

    def norm_plot_fn(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
        if path:
            x = np.array(path.x)
            norm_obs = x[..., :obs_dim]
            action = x[..., obs_dim:]
            unnorm_action = norm_env.unnormalise_action(action)
            unnorm_obs = norm_env.unnormalise_obs(norm_obs)
            unnorm_x = np.concatenate([unnorm_obs, unnorm_action], axis=-1)
            try:
                y = path.y
                unnorm_y = norm_env.unnormalise_obs(y)
            except AttributeError:
                pass
            path = PlotTuple(x=unnorm_x, y=unnorm_y)
        return plot_fn(path, ax=ax, fig=fig, domain=unnorm_domain, path_str=path_str, env=env)

    return norm_plot_fn
