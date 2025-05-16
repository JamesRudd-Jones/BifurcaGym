import jax
import jax.random as jrandom
import jax.numpy as jnp

import pytest

import chex

import project_name

import itertools
import copy


env_names = ["Pendulum-v0",
             ]
cont_state = [True]#, False]
cont_action = [True]#, False]
normalised = [True]
delta_obs = [True]
autoreset = [True]

all_combinations = list(itertools.product(env_names,
                                          cont_state,
                                          cont_action,
                                          normalised,
                                          delta_obs,
                                          autoreset,
                                          ))

@pytest.mark.parametrize("env_name, "
                         "cont_state, "
                         "cont_action, "
                         "normalised, "
                         "delta_obs, "
                         "autoreset, ",
                         all_combinations)


class TestWrapper:
    def setup_method(self):
        """Set up common test resources."""
        self.num_steps = 10#0#0
        self.num_episodes = 100
        self.key = jrandom.PRNGKey(42)
        self.error = 1e-4

    # def test_reset(self, gym_env_name):
    #     """Test auto reset works"""
    #     # env_gym = gym.make(env_name)
    #     key = jrandom.PRNGKey(0)
    #     env_jax, env_params = project_name.make(gym_env_name)
    #     for _ in range(num_episodes):
    #         key, _key = jrandom.split(key)
    #         obs, state = env_jax.reset(_key, env_params)
    #         # Check state and observation space
    #         env_jax.state_space(env_params).contains(state)
    #         env_jax.observation_space(env_params).contains(obs)

    def _test_normalised_obs(self, wrapped_env, obs, w_obs):
        unnorm_obs = wrapped_env.unnormalise_obs(w_obs)
        renorm_obs = wrapped_env.normalise_obs(unnorm_obs)
        chex.assert_trees_all_close(w_obs, renorm_obs, atol=self.error)

        chex.assert_trees_all_close(unnorm_obs, obs, atol=self.error)

    def _test_delta_obs_normalised(self, wrapped_env, obs, w_obs, nobs):
        w_nobs = obs + wrapped_env.unnormalise_obs(w_obs)
        chex.assert_trees_all_close(w_nobs, nobs, atol=self.error)

    def _test_delta_obs(self, wrapped_env, obs, w_obs, nobs):
        w_nobs = obs + w_obs
        chex.assert_trees_all_close(w_nobs, nobs, atol=self.error)

    def _test_rew_fn(self, gt_rew, old_obs, action, obs, wrapped_env, key):
        x = jnp.concatenate([old_obs, action])
        y = obs
        norm_rew = wrapped_env.reward_function(x, y, key)
        chex.assert_trees_all_close(gt_rew, norm_rew, atol=self.error)
        # assert jnp.allclose(gt_rew, norm_rew), f"gt_rew: {gt_rew}, norm_rew: {norm_rew}"

    def _test_update_function(self, start_obs, action, delta_obs, next_obs, update_fn, env, env_params):
        x = jnp.concatenate([start_obs, action], axis=-1)
        updated_next_obs = update_fn(x, delta_obs, env, env_params)
        chex.assert_trees_all_close(next_obs, updated_next_obs, atol=self.error)
        # assert jnp.allclose(next_obs,
        #                     updated_next_obs), f"Next obs: {next_obs} and updated next obs: {updated_next_obs}"

    def test_normal(self, env_name, cont_state, cont_action, normalised, delta_obs, autoreset):
        try:
            key, _key = jrandom.split(self.key)
            env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                    normalised=False, delta_obs=False, autoreset=False)
            wrapped_env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                            normalised=normalised, delta_obs=delta_obs, autoreset=autoreset)
            total_rew = 0
            observations = []
            next_observations = []
            rewards = []
            actions = []
            # Loop over test episodes
            for _ in range(self.num_episodes):
                obs, env_state = env.reset(_key)
                w_obs, w_env_state = wrapped_env.reset(_key)
                if normalised and not delta_obs:
                    self._test_normalised_obs(wrapped_env, obs, w_obs)
                for _ in range(self.num_steps):
                    # observations.append(obs)
                    key, _key = jrandom.split(key)
                    action = env.action_space().sample(_key)
                    w_action = wrapped_env.action_space().sample(_key)
                    # actions.append(action)
                    key, _key = jrandom.split(key)
                    nobs, env_state, rew, done, info = env.step(action, env_state, _key)
                    w_obs, w_env_state, w_rew, w_done, w_info = wrapped_env.step(w_action, w_env_state, _key)
                    # next_observations.append(obs)
                    # total_rew += rew
                    # rewards.append(rew)
                    if normalised and not delta_obs:
                        self._test_normalised_obs(wrapped_env, nobs, w_obs)
                    if delta_obs and not normalised:
                        self._test_delta_obs(wrapped_env, obs, w_obs, nobs)
                    if delta_obs and normalised:
                        self._test_delta_obs_normalised(wrapped_env, obs, w_obs, nobs)
                    # key, _key = jrandom.split(key)
                    # self._test_rew_fn(rew, old_obs, action, obs, wrapped_env, _key)

                    obs = copy.deepcopy(nobs)

                    if done:
                        break
                # observations = jnp.array(observations)
                # actions = jnp.array(actions)
                # rewards = jnp.array(rewards)
                # next_observations = jnp.array(next_observations)
                # teleport_deltas = jnp.array(teleport_deltas)
                # x = jnp.concatenate([observations, actions], axis=1)
                # teleport_next_obs = teleport_update_fn(x, teleport_deltas, wrapped_env, env_params)
                # assert jnp.allclose(teleport_next_obs, next_observations)
                # test_rewards = wrapped_env.reward_function(x, next_observations, env_params)
                # assert jnp.allclose(rewards, test_rewards), f"Rewards: {rewards} not equal to test rewards: {test_rewards}"
                # print(f"passed!, rew={total_rew}")

        except ValueError as e:
            print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test_generative_step for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    def test_genstep(self, env_name, cont_state, cont_action, normalised, delta_obs, autoreset):
        try:
            key, _key = jrandom.split(self.key)
            env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                    normalised=False, delta_obs=False, autoreset=False)
            wrapped_env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                            normalised=normalised, delta_obs=delta_obs, autoreset=autoreset)
            total_rew = 0
            observations = []
            next_observations = []
            rewards = []
            actions = []
            # Loop over test episodes
            for _ in range(self.num_episodes):
                obs, env_state = env.reset(_key)
                w_obs, w_env_state = wrapped_env.reset(_key)
                if normalised and not delta_obs:
                    self._test_normalised_obs(wrapped_env, obs, w_obs)
                for _ in range(self.num_steps):
                    # observations.append(obs)
                    key, _key = jrandom.split(key)
                    action = env.action_space().sample(_key)
                    w_action = wrapped_env.action_space().sample(_key)
                    # actions.append(action)
                    key, _key = jrandom.split(key)
                    nobs, env_state, rew, done, info = env.step(action, env_state, _key)
                    w_obs, w_env_state, w_rew, w_done, w_info = wrapped_env.generative_step(w_action, obs, _key)
                    # TODO sort out generative step with wrappers as it is not working atm
                    # next_observations.append(obs)
                    # total_rew += rew
                    # rewards.append(rew)
                    if normalised and not delta_obs:
                        self._test_normalised_obs(wrapped_env, nobs, w_obs)
                    if delta_obs and not normalised:
                        self._test_delta_obs(wrapped_env, obs, w_obs, nobs)
                    if delta_obs and normalised:
                        self._test_delta_obs_normalised(wrapped_env, obs, w_obs, nobs)
                    # key, _key = jrandom.split(key)
                    # self._test_rew_fn(rew, old_obs, action, obs, wrapped_env, _key)

                    obs = copy.deepcopy(nobs)

                    if done:
                        break
                # observations = jnp.array(observations)
                # actions = jnp.array(actions)
                # rewards = jnp.array(rewards)
                # next_observations = jnp.array(next_observations)
                # teleport_deltas = jnp.array(teleport_deltas)
                # x = jnp.concatenate([observations, actions], axis=1)
                # teleport_next_obs = teleport_update_fn(x, teleport_deltas, wrapped_env, env_params)
                # assert jnp.allclose(teleport_next_obs, next_observations)
                # test_rewards = wrapped_env.reward_function(x, next_observations, env_params)
                # assert jnp.allclose(rewards, test_rewards), f"Rewards: {rewards} not equal to test rewards: {test_rewards}"
                # print(f"passed!, rew={total_rew}")

        except ValueError as e:
            print(
                f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(
                f"Unexpected error during test_generative_step for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    # TODO add in test autoreset vs something else, perhaps do a usual break if done and then an autoreset one
