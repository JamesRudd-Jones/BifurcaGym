import jax
import jax.numpy as jnp
from functools import partial
import chex
from typing import Any, Dict, Tuple, Union
from bifurcagym.envs import EnvState, EnvParams
from bifurcagym import spaces


class NormalisedEnvCSDA(object):
    """
    Normalises obs and rewards
    """
    def __init__(self, wrapped_normalised_env):

        self._wrapped_normalised_env = wrapped_normalised_env
        self.unnorm_action_space = self._wrapped_normalised_env.action_space
        self.unnorm_obs_space = self._wrapped_normalised_env.observation_space

        # self.unnorm_obs_space_range = self.unnorm_obs_space.high - self.unnorm_obs_space.low

        # self.norm_obs_space_range = self.observation_space().high - self.observation_space().low

    @property
    def wrapped_normalised_env(self):
        return self._wrapped_normalised_env

    def unnorm_obs_space_range(self, params: EnvParams):
        return self.unnorm_obs_space(params).high - self.unnorm_obs_space(params).low

    def norm_obs_space_range(self, params: EnvParams):
        return self.observation_space(params).high - self.observation_space(params).low

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: chex.Numeric,
             state: EnvState,
             params: EnvParams,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        unnorm_obs, unnorm_delta_obs, env_state, rew, done, info = self._wrapped_normalised_env.step(action,
                                                                                                     state,
                                                                                                     params,
                                                                                                     key)

        return (self.normalise_obs(unnorm_obs, params),
                self.normalise_delta_obs(unnorm_delta_obs, params),
                env_state,
                rew,
                done,
                info)

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: chex.Numeric,
                        gen_obs: chex.Array,
                        params: EnvParams,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        return self.step(action, self.get_state(gen_obs, params), params, key)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        unnorm_obs, env_state = self._wrapped_normalised_env.reset(params, key)
        return self.normalise_obs(unnorm_obs, params), env_state

    @partial(jax.jit, static_argnums=(0,))
    def apply_delta_obs(self, obs: chex.Array, delta_obs: chex.Array, params: EnvParams) -> chex.Array:
        unnorm_nobs = self._wrapped_normalised_env.apply_delta_obs(self.unnormalise_obs(obs, params),
                                                                   self.unnormalise_delta_obs(delta_obs, params),
                                                                   params)
        return self.normalise_obs(unnorm_nobs, params)

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return self._wrapped_normalised_env.get_state(self.unnormalise_obs(obs, params), params)

    def normalise_obs(self, obs: chex.Array, params: EnvParams) -> chex.Array:
        pos_obs = obs - self.unnorm_obs_space(params).low
        norm_obs = self.norm_obs_space_range(params) * pos_obs / self.unnorm_obs_space_range(params) + self.observation_space(params).low

        return norm_obs

    def normalise_delta_obs(self, obs: chex.Array, params: EnvParams) -> chex.Array:
        norm_obs = obs / self.unnorm_obs_space_range(params) * 2  # TODO the original states times by two, unsure this is good for true normalisation

        return norm_obs

    def unnormalise_obs(self, norm_obs: chex.Array, params: EnvParams) -> chex.Array:
        pos_obs = self.unnorm_obs_space_range(params) * (norm_obs - self.observation_space(params).low) / self.norm_obs_space_range(params)
        obs = pos_obs + self.unnorm_obs_space(params).low

        return obs

    def unnormalise_delta_obs(self, obs: chex.Array, params: EnvParams) -> chex.Array:
        range = self.unnorm_obs_space_range(params)
        unnorm_obs = obs * range / 2  # TODO see above regarding the original reference

        return unnorm_obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(low=-jnp.ones_like(self.unnorm_obs_space(params)),
                          high=jnp.ones_like(self.unnorm_obs_space(params)),
                          shape=self.unnorm_obs_space(params).shape,
                          dtype=self.unnorm_obs_space(params).dtype)

    def __getattr__(self, attr):
        if attr == "_wrapped_normalised_env":
            raise AttributeError()
        return getattr(self._wrapped_normalised_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_normalised_env)


class NormalisedEnvCSCA(NormalisedEnvCSDA):
    """
    Normalises obs and rewards and actions
    """
    def __init__(self, wrapped_normalised_env):
        super().__init__(wrapped_normalised_env)

    def unnorm_action_space_range(self, params: EnvParams):
        return self.unnorm_action_space(params).high - self.unnorm_action_space(params).low

    def norm_action_space_range(self, params: EnvParams):
        return self.action_space(params).high - self.action_space(params).low

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: chex.Numeric,
             state: EnvState,
             params: EnvParams,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        unnorm_action = self.unnormalise_action(action, params)
        unnorm_obs, unnorm_delta_obs, env_state, rew, done, info = self._wrapped_normalised_env.step(unnorm_action,
                                                                                                     state,
                                                                                                     params,
                                                                                                     key)

        return (self.normalise_obs(unnorm_obs, params),
                self.normalise_delta_obs(unnorm_delta_obs, params),
                env_state,
                rew,
                done,
                info)

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> chex.Array:
        unnorm_action = self.unnormalise_action(input_action_t, params)

        return self._wrapped_normalised_env.reward_and_done_function(unnorm_action,
                                                                     state_t,
                                                                     state_tp1,
                                                                     params,
                                                                     key)

    def normalise_action(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        pos_action = action - self.unnorm_action_space(params).low
        norm_action = self.norm_action_space_range(params) * pos_action / self.unnorm_action_space_range(params) + self.action_space(params).low

        return norm_action

    def unnormalise_action(self, norm_action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        pos_action = self.unnorm_action_space_range(params) * (norm_action - self.action_space(params).low) / self.norm_action_space_range(params)
        action = pos_action + self.unnorm_action_space(params).low

        return action

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(low=-jnp.ones_like(self.unnorm_action_space(params)),
                          high=jnp.ones_like(self.unnorm_action_space(params)),
                          shape=self.unnorm_action_space(params).shape,
                          dtype=self.unnorm_action_space(params).dtype)
