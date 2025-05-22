import jax
import jax.numpy as jnp
from functools import partial
import chex
from typing import Any, Dict, Tuple, Union
from bifurcagym.envs import EnvState
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
    Normalises obs to be between -1 and 1
    """
    def __init__(self, wrapped_normalised_env):

        self._wrapped_normalised_env = wrapped_normalised_env
        self.unnorm_action_space = self._wrapped_normalised_env.action_space()
        self.unnorm_observation_space = self._wrapped_normalised_env.observation_space()

        self.unnorm_obs_space_size = self.unnorm_observation_space.high - self.unnorm_observation_space.low

    @property
    def wrapped_normalised_env(self):
        return self._wrapped_normalised_env

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[jnp.int_, jnp.float_, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        unnorm_obs, unnorm_delta_obs, new_env_state, rew, done, info = self._wrapped_normalised_env.step(action,
                                                                                                         state,
                                                                                                         key)

        return self.normalise_obs(unnorm_obs), self.normalise_delta_obs(unnorm_delta_obs), new_env_state, rew, done, info

    @partial(jax.jit, static_argnums=(0,))
    def generative_step(self,
                        action: Union[jnp.int_, jnp.float_, chex.Array],
                        gen_obs: chex.Array,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        return self.step(action, self.get_state(gen_obs), key)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        unnorm_obs, env_state = self._wrapped_normalised_env.reset(key)
        return self.normalise_obs(unnorm_obs), env_state

    @partial(jax.jit, static_argnums=(0,))
    def apply_delta_obs(self, obs: chex.Array, delta_obs: chex.Array) -> chex.Array:
        unnorm_nobs = self._wrapped_normalised_env.apply_delta_obs(self.unnormalise_obs(obs),
                                                                   self.unnormalise_delta_obs(delta_obs))
        return self.normalise_obs(unnorm_nobs)

    def get_state(self, obs: chex.Array) -> EnvState:
        return self._wrapped_normalised_env.get_state(self.unnormalise_obs(obs))

    def normalise_obs(self,  obs: chex.Array) -> chex.Array:
        low = self.unnorm_observation_space.low
        size = self.unnorm_obs_space_size
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1

        return norm_obs

    def normalise_delta_obs(self,  obs: chex.Array) -> chex.Array:
        size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        norm_obs = obs / size * 2  # TODO the original states times by two, unsure this is good for true normalisation

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
        unnorm_obs = obs * size / 2  # TODO see above regarding the original reference

        return unnorm_obs

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1,
                          high=1,
                          shape=self.unnorm_observation_space.shape,
                          dtype=self.unnorm_observation_space.dtype)

    def __getattr__(self, attr):
        if attr == "_wrapped_normalised_env":
            raise AttributeError()
        return getattr(self._wrapped_normalised_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_normalised_env)


class NormalisedEnvCSCA(NormalisedEnvCSDA):
    """
    Normalises obs to be between -1 and 1
    Normalises actions to be between -1 and 1
    """
    def __init__(self, wrapped_normalised_env):
        super().__init__(wrapped_normalised_env)

        self.unnorm_action_space_size = self.unnorm_action_space.high - self.unnorm_action_space.low

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: Union[jnp.int_, jnp.float_, chex.Array],
             state: EnvState,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        unnorm_action = self.unnormalise_action(action)
        unnorm_obs, unnorm_delta_obs, new_env_state, rew, done, info = self._wrapped_normalised_env.step(unnorm_action,
                                                                                                  state,
                                                                                                  key)

        return self.normalise_obs(unnorm_obs), self.normalise_delta_obs(unnorm_delta_obs), new_env_state, rew, done, info

    def reward_function(self,
                        input_action_t: Union[int, float, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey,
                        ) -> chex.Array:
        unnorm_action = self.unnormalise_action(input_action_t)

        return self._wrapped_normalised_env.reward_function(unnorm_action, state_t, state_tp1, key)

    def unnormalise_action(self, action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        low = self.unnorm_action_space.low
        size = self.unnorm_action_space_size
        act01 = (action + 1) / 2
        act_ranged = act01 * size
        unnorm_act = act_ranged + low

        return unnorm_act

    def normalise_action(self, action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        low = self.unnorm_action_space.low
        size = self.unnorm_action_space_size
        pos_action = action - low
        norm_action = (pos_action / size * 2) - 1

        return norm_action

    def action_space(self) -> spaces.Box:
        return spaces.Box(low=-1,
                          high=1,
                          shape=self.unnorm_action_space.shape,
                          dtype=self.unnorm_action_space.dtype)
