import jax
import jax.numpy as jnp
from functools import partial
import chex
from typing import Any, Dict, Tuple, Union
from bifurcagym.envs import EnvState, EnvParams
from bifurcagym import spaces


# def make_normalised_plot_fn(norm_env, env_params, plot_fn):
#     obs_dim = norm_env.observation_space().low.size
#     wrapped_env = norm_env.wrapped_env
#     # Set domain
#     low = np.concatenate([wrapped_env.observation_space(env_params).low,
#                           np.expand_dims(np.array(wrapped_env.action_space(env_params).low), axis=0)])
#     high = np.concatenate([wrapped_env.observation_space(env_params).high,
#                            np.expand_dims(np.array(wrapped_env.action_space(env_params).high), axis=0)])
#     unnorm_domain = [elt for elt in zip(low, high)]
#
#     def norm_plot_fn(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
#         if path:
#             x = np.array(path.x)
#             norm_obs = x[..., :obs_dim]
#             action = x[..., obs_dim:]
#             unnorm_action = norm_env.unnormalise_action(action)
#             unnorm_obs = norm_env.unnormalise_obs(norm_obs)
#             unnorm_x = np.concatenate([unnorm_obs, unnorm_action], axis=-1)
#             try:
#                 y = path.y
#                 unnorm_y = norm_env.unnormalise_obs(y)
#             except AttributeError:
#                 pass
#             path = PlotTuple(x=unnorm_x, y=unnorm_y)
#         return plot_fn(path, ax=ax, fig=fig, domain=unnorm_domain, path_str=path_str, env=env)
#
#     return norm_plot_fn


class NormalisedEnvCSDA(object):
    """
    Normalises obs and rewards
    """
    def __init__(self, wrapped_normalised_env):

        self._wrapped_normalised_env = wrapped_normalised_env
        self.unnorm_action_space = self._wrapped_normalised_env.action_space()
        self.unnorm_obs_space = self._wrapped_normalised_env.observation_space()
        self.unnorm_rew_space = self._wrapped_normalised_env.reward_space()

        self.unnorm_obs_space_range = self.unnorm_obs_space.high - self.unnorm_obs_space.low
        self.unnorm_rew_space_range = self.unnorm_rew_space.high - self.unnorm_rew_space.low

        self.norm_obs_space_range = self.observation_space().high - self.observation_space().low
        self.norm_rew_space_range = self.reward_space().high - self.reward_space().low

    @property
    def wrapped_normalised_env(self):
        return self._wrapped_normalised_env

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

        return (self.normalise_obs(unnorm_obs),
                self.normalise_delta_obs(unnorm_delta_obs),
                env_state,
                self.normalise_rew(rew),
                done,
                info)

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: chex.Numeric,
                        gen_obs: chex.Array,
                        params: EnvParams,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        return self.step(action, self.get_state(gen_obs), params, key)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        unnorm_obs, env_state = self._wrapped_normalised_env.reset(params, key)
        return self.normalise_obs(unnorm_obs), env_state

    @partial(jax.jit, static_argnums=(0,))
    def apply_delta_obs(self, obs: chex.Array, delta_obs: chex.Array) -> chex.Array:
        unnorm_nobs = self._wrapped_normalised_env.apply_delta_obs(self.unnormalise_obs(obs),
                                                                   self.unnormalise_delta_obs(delta_obs))
        return self.normalise_obs(unnorm_nobs)

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> chex.Array:
        reward = self._wrapped_normalised_env.reward_and_done_function(input_action_t, state_t, state_tp1, params, key)
        return self.normalise_rew(reward)

    def get_state(self, obs: chex.Array) -> EnvState:
        return self._wrapped_normalised_env.get_state(self.unnormalise_obs(obs))

    def normalise_obs(self,  obs: chex.Array) -> chex.Array:
        pos_obs = obs - self.unnorm_obs_space.low
        norm_obs = self.norm_obs_space_range * pos_obs / self.unnorm_obs_space_range + self.observation_space().low

        return norm_obs

    def normalise_delta_obs(self,  obs: chex.Array) -> chex.Array:
        norm_obs = obs / self.unnorm_obs_space_range * 2  # TODO the original states times by two, unsure this is good for true normalisation

        return norm_obs

    def normalise_rew(self,  rew: chex.Array) -> chex.Array:
        pos_rew = rew - self.unnorm_rew_space.low
        norm_rew = self.norm_rew_space_range * pos_rew / self.unnorm_rew_space_range + self.reward_space().low

        return norm_rew

    def unnormalise_obs(self, norm_obs: chex.Array) -> chex.Array:
        pos_obs = self.unnorm_obs_space_range * (norm_obs - self.observation_space().low) / self.norm_obs_space_range
        obs = pos_obs + self.unnorm_obs_space.low

        return obs

    def unnormalise_delta_obs(self, obs: chex.Array) -> chex.Array:
        range = self.unnorm_obs_space_range
        unnorm_obs = obs * range / 2  # TODO see above regarding the original reference

        return unnorm_obs

    def unnormalise_rew(self, norm_rew: chex.Array) -> chex.Array:
        pos_rew = self.unnorm_rew_space_range * (norm_rew - self.reward_space().low) / self.norm_rew_space_range
        rew = pos_rew + self.unnorm_rew_space.low

        return rew

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-jnp.ones_like(self.unnorm_obs_space),
                          high=jnp.ones_like(self.unnorm_obs_space),
                          shape=self.unnorm_obs_space.shape,
                          dtype=self.unnorm_obs_space.dtype)

    def reward_space(self) -> spaces.Box:
        return spaces.Box(low=-jnp.ones_like(self.unnorm_rew_space),
                          high=jnp.zeros_like(self.unnorm_rew_space),
                          shape=self.unnorm_rew_space.shape,
                          dtype=self.unnorm_rew_space.dtype)
    # TODO what do we want the normalisation of reward to be? Currently to -1 and 1, is that okay?

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

        self.unnorm_action_space_range = self.unnorm_action_space.high - self.unnorm_action_space.low

        self.norm_action_space_range = self.action_space().high - self.action_space().low

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: chex.Numeric,
             state: EnvState,
             params: EnvParams,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, unnorm_delta_obs, env_state, rew, done, info = self._wrapped_normalised_env.step(unnorm_action,
                                                                                                     state,
                                                                                                     params,
                                                                                                     key)

        return (self.normalise_obs(unnorm_obs),
                self.normalise_delta_obs(unnorm_delta_obs),
                env_state,
                self.normalise_rew(rew),
                done,
                info)

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> chex.Array:
        unnorm_action = self.unnormalise_action(input_action_t)
        unnorm_reward = self._wrapped_normalised_env.reward_and_done_function(unnorm_action,
                                                                              state_t,
                                                                              state_tp1,
                                                                              params,
                                                                              key)

        return self.normalise_rew(unnorm_reward)

    def normalise_action(self, action: chex.Numeric) -> chex.Numeric:
        pos_action = action - self.unnorm_action_space.low
        norm_action = self.norm_action_space_range * pos_action / self.unnorm_action_space_range + self.action_space().low

        return norm_action

    def unnormalise_action(self, norm_action: chex.Numeric) -> chex.Numeric:
        pos_action = self.unnorm_action_space_range * (norm_action - self.action_space().low) / self.norm_action_space_range
        action = pos_action + self.unnorm_action_space.low

        return action

    def action_space(self) -> spaces.Box:
        return spaces.Box(low=-jnp.ones_like(self.unnorm_action_space_range),
                          high=jnp.ones_like(self.unnorm_action_space_range),
                          shape=self.unnorm_action_space.shape,
                          dtype=self.unnorm_action_space.dtype)
