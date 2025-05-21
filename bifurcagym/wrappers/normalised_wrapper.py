from gymnax.environments import spaces
from copy import deepcopy
import jax.numpy as jnp
import jax.random as jrandom
import jax
from functools import partial
import chex
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, overload
from bifurcagym.envs import base_env
from bifurcagym.envs.base_env import EnvState


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
    Normalises obs to be between -1 and 1
    """
    def __init__(self, wnormalised_env):

        self._wnormalised_env = wnormalised_env
        self.unnorm_action_space = self._wnormalised_env.action_space()
        self.unnorm_observation_space = self._wnormalised_env.observation_space()

        self.unnorm_obs_space_size = self.unnorm_observation_space.high - self.unnorm_observation_space.low

    # @property
    # def wnormalised_env(self):
    #     return self._wnormalised_env

    def action_space(self) -> spaces.Discrete:
        return self.unnorm_action_space

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-jnp.ones_like(self.unnorm_observation_space.low,),
                          high=jnp.ones_like(self.unnorm_observation_space.high,),
                          shape=self.unnorm_observation_space.shape,
                          dtype=self.unnorm_observation_space.dtype)

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[int, float, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        unnorm_obs, unnorm_delta_obs, new_env_state, rew, done, info = self._wnormalised_env.step(action, state, key)

        return self.normalise_obs(unnorm_obs), self.normalise_delta_obs(unnorm_delta_obs), new_env_state, rew, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: Union[int, float, chex.Array],
                        norm_obs: chex.Array,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        unnorm_init_obs = self.unnormalise_obs(norm_obs)
        return self.step(action, self._wnormalised_env.get_state(unnorm_init_obs), key)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        unnorm_obs, env_state = self._wnormalised_env.reset(key)
        return self.normalise_obs(unnorm_obs), env_state

    # TODO might need a render normalise as well

    def __getattr__(self, attr):
        if attr == "_wnormalised_env":
            raise AttributeError()
        return getattr(self._wnormalised_env, attr)

    # def __str__(self):
    #     return "{}({})".format(type(self).__name__, self.wnormalised_env)

    def normalise_obs(self,  obs: chex.Array) -> chex.Array:
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1

        return norm_obs

    def normalise_delta_obs(self,  obs: chex.Array) -> chex.Array:
        size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        norm_obs = obs / size # * 2  # TODO the original states times by two, unsure this is good for true normalisation

        return norm_obs

    def unnormalise_obs(self, obs: chex.Array) -> chex.Array:
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        obs01 = (obs + 1) / 2
        obs_ranged = obs01 * size
        unnorm_obs = obs_ranged + low

        return unnorm_obs

    def unnormalise_delta_obs(self, obs: chex.Array) -> chex.Array:
        size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        unnorm_obs = obs * size   # TODO see above regarding the original reference

        return unnorm_obs


class NormalisedEnvCSCA(NormalisedEnvCSDA):
    """
    Normalises obs to be between -1 and 1
    Normalises actions to be between -1 and 1
    """
    def __init__(self, wnormalised_env):
        super().__init__(wnormalised_env)

        self.unnorm_action_space_size = self.unnorm_action_space.high - self.unnorm_action_space.low

    def action_space(self) -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=self.unnorm_action_space.shape, dtype=self.unnorm_action_space.dtype)

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[int, float, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, unnorm_delta_obs, new_env_state, rew, done, info = self._wnormalised_env.step(unnorm_action,
                                                                                                  state,
                                                                                                  key)

        return self.normalise_obs(unnorm_obs), self.normalise_delta_obs(unnorm_delta_obs), new_env_state, rew, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: Union[int, float, chex.Array],
                        norm_obs: chex.Array,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        unnorm_init_obs = self.unnormalise_obs(norm_obs)
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, unnorm_delta_obs, new_env_state, rew, done, info = self._wnormalised_env.generative_step(unnorm_action,
                                                                                           unnorm_init_obs,
                                                                                           key)

        return self.normalise_obs(unnorm_obs), self.normalise_delta_obs(unnorm_delta_obs), new_env_state, rew, done, info

    def reward_func(self,
                    action_t: Union[int, float, chex.Array],
                    state_t: EnvState,
                    state_tp1: EnvState,
                    key: chex.PRNGKey,
                    ) -> chex.Array:
        unnorm_action = self.unnormalise_action(action_t)
        rewards = self._wnormalised_env.reward_func(unnorm_action, state_t, state_tp1, key)

        return rewards

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


class NormalisedEnvDeltaObsCSDA(NormalisedEnvCSDA):
    """
    A work-around for the normalisation being different if using delta_obs
    """

    def __init__(self, wnormalised_env):
        super().__init__(wnormalised_env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        return self._wnormalised_env.reset(key)
    # TODO we don't have a delta obs at reset so this be pointless to normalise right?

    def normalise_obs(self,  obs: chex.Array) -> chex.Array:
        size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        norm_obs = obs / size

        return norm_obs

    # We don't need an unnormalise obs to be different as the input for generative step is for normalised values
    # However we do need the below for test verification

    def test_normalise_obs(self,  obs: chex.Array) -> chex.Array:
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1

        return norm_obs

    def test_unnormalise_obs(self, obs: chex.Array) -> chex.Array:
        size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        unnorm_obs = obs * size

        return unnorm_obs


class NormalisedEnvDeltaObsCSCA(NormalisedEnvCSCA):
    """
    A work-around for the normalisation being different if using delta_obs
    """

    def __init__(self, wnormalised_env):
        super().__init__(wnormalised_env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        return self._wnormalised_env.reset(key)
    # TODO we don't have a delta obs at reset so this be pointless to normalise right?

    def normalise_obs(self, obs: chex.Array) -> chex.Array:
        size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        norm_obs = obs / size

        return norm_obs

        # We don't need an unnormalise obs to be different as the input for generative step is for normalised values
        # However we do need the below for test verification

    def test_normalise_obs(self,  obs: chex.Array) -> chex.Array:
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1

        return norm_obs

    def test_unnormalise_obs(self, obs: chex.Array) -> chex.Array:
        size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        unnorm_obs = obs * size

        return unnorm_obs