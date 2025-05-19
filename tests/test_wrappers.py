import jax
import jax.random as jrandom
import jax.numpy as jnp

import pytest

import chex

import project_name

import itertools
import copy


env_names = ["KS-v0",
             ]
cont_state = [True]#, False]
cont_action = [True]#, False]
normalised = [True, False]
delta_obs = [False]#, True]
autoreset = [True, False]

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
        self.num_episodes = 10#0
        self.key = jrandom.PRNGKey(42)
        self.error = 1e-4

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

    def _test_update_function(self, start_obs, action, delta_obs, next_obs, update_fn, env, env_params):
        x = jnp.concatenate([start_obs, action], axis=-1)
        updated_next_obs = update_fn(x, delta_obs, env, env_params)
        chex.assert_trees_all_close(next_obs, updated_next_obs, atol=self.error)

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
                    if normalised:
                        w_obs, w_env_state, w_rew, w_done, w_info = wrapped_env.generative_step(w_action, wrapped_env.normalise_obs(obs), _key)
                    else:
                        w_obs, w_env_state, w_rew, w_done, w_info = wrapped_env.generative_step(w_action, obs, _key)
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

    def test_autoreset(self, env_name, cont_state, cont_action, normalised, delta_obs, autoreset):
        try:
            key, _key = jrandom.split(self.key)
            env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                    normalised=False, delta_obs=False, autoreset=False)
            wrapped_env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                            normalised=normalised, delta_obs=delta_obs, autoreset=autoreset)
            observations = []
            nobservations = []
            rewards = []
            actions = []
            dones = []
            # Loop over test episodes
            key = self.key
            key, _key = jrandom.split(key)
            obs, env_state = env.reset(_key)
            for _ in range(self.num_episodes):
                for _ in range(self.num_steps):
                    key, _key = jrandom.split(key)
                    action = env.action_space().sample(_key)
                    key, _key = jrandom.split(key)
                    nobs, env_state, rew, done, info = env.step(action, env_state, _key)
                    # _key is 3122727659, 270479714
                    observations.append(obs)
                    nobservations.append(nobs)
                    actions.append(action)
                    rewards.append(rew)
                    dones.append(done)

                    obs = copy.deepcopy(nobs)

                    if done:
                        _, reset_key = jrandom.split(_key)
                        # reset_key is 3320941097, 3018999918
                        obs, env_state = env.reset(reset_key)
                        break

            obs = jnp.array(observations)
            nobs = jnp.array(nobservations)
            actions = jnp.array(actions)
            rewards = jnp.array(rewards)
            dones = jnp.array(dones)

            def scan_step(carry, _):
                state, obs, key = carry
                key, _key = jrandom.split(key)
                action = wrapped_env.action_space().sample(_key)
                key, _key = jrandom.split(key)
                nobs, next_state, reward, done, info = wrapped_env.step(action, state, _key)
                return (next_state, nobs, key), (obs, nobs, action, reward, done)

            key, _key = jrandom.split(self.key)
            init_w_obs, w_env_state = wrapped_env.reset(_key)
            with jax.disable_jit(disable=False):
                (final_state, _, _), (w_obs, w_nobs, w_actions, w_rewards, w_dones) = jax.lax.scan(scan_step, (w_env_state, init_w_obs, key), None, self.num_steps * self.num_episodes)

            obs_length = len(obs)

            w_obs = w_obs[:obs_length]
            w_nobs = w_nobs[:obs_length]
            w_actions = w_actions[:obs_length]
            w_rewards = w_rewards[:obs_length]
            w_dones = w_dones[:obs_length]

            if normalised and not delta_obs:
                self._test_normalised_obs(wrapped_env, nobs, w_obs)
            if delta_obs and not normalised:
                self._test_delta_obs(wrapped_env, obs, w_obs, nobs)
            if delta_obs and normalised:
                self._test_delta_obs_normalised(wrapped_env, obs, w_obs, nobs)

            if not cont_state:
                chex.assert_trees_all_equal(w_obs, obs)

            if normalised:
                chex.assert_trees_all_close(wrapped_env.unnormalise_action(w_actions), actions, atol=self.error)
            else:
                if cont_action:
                    chex.assert_trees_all_close(w_actions, actions, atol=self.error)
                else:
                    chex.assert_trees_all_equal(w_actions, actions)

            chex.assert_trees_all_close(w_rewards, rewards, atol=self.error)
            chex.assert_trees_all_equal(w_dones, dones)

        except ValueError as e:
            print(
                f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(
                f"Unexpected error during test_generative_step for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
