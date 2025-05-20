import jax
import jax.random as jrandom
import jax.numpy as jnp

import pytest

import chex

import project_name

import itertools
import copy


env_names = [
             # "Acrobot-v0",
             "Pendulum-v0",
             "PilcoCartPole-v0",
             # "HenonMap-v0",
             "LogisticMap-v0",
             "TentMap-v0",
             ]
cont_state = [True, False]
cont_action = [True, False]
normalised = [True, False]
delta_obs = [True, False]

all_combinations = list(itertools.product(env_names,
                                          cont_state,
                                          cont_action,
                                          normalised,
                                          delta_obs,
                                          ))

@pytest.mark.parametrize("env_name, "
                         "cont_state, "
                         "cont_action, "
                         "normalised, "
                         "delta_obs, ",
                         all_combinations)


class TestWrapper:
    def setup_method(self):
        """Set up common test resources."""
        self.num_steps = 100#0
        self.num_episodes = 10#0
        self.key = jrandom.PRNGKey(42)
        self.error = 1e-4

    def _test_normalised_obs(self, wrapped_env, obs, w_obs):
        unnorm_obs = wrapped_env.unnormalise_obs(w_obs)
        renorm_obs = wrapped_env.normalise_obs(unnorm_obs)
        chex.assert_trees_all_close(w_obs, renorm_obs, atol=self.error)

        chex.assert_trees_all_close(unnorm_obs, obs, atol=self.error)

    def _test_delta_obs(self, wrapped_env, obs, nobs, w_obs, normalised):
        if normalised:
            w_nobs = obs + wrapped_env.test_unnormalise_obs(w_obs)
        else:
            w_nobs = obs + w_obs
        chex.assert_trees_all_close(w_nobs, nobs, atol=self.error)

    def _test_rew_fn(self, reward_t, action_t, state_t, state_tp1, w_reward_t, w_action_t, w_state_t, w_state_tp1,
                     env, wrapped_env, key):
        reward = env.reward_func(action_t, state_t, state_tp1, key)
        w_reward = wrapped_env.reward_func(w_action_t, w_state_t, w_state_tp1, key)
        chex.assert_trees_all_close(reward, reward_t, atol=self.error)
        chex.assert_trees_all_close(w_reward, w_reward_t, atol=self.error)
        chex.assert_trees_all_close(reward_t, w_reward_t, atol=self.error)

    def test_normal(self, env_name, cont_state, cont_action, normalised, delta_obs):
        try:
            key, _key = jrandom.split(self.key)
            env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                    normalised=False, delta_obs=False, autoreset=False)
            wrapped_env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                            normalised=normalised, delta_obs=delta_obs, autoreset=False)
            # Loop over test episodes
            for _ in range(self.num_episodes):
                obs, env_state = env.reset(_key)
                w_obs, w_env_state = wrapped_env.reset(_key)
                if normalised and not delta_obs:
                    self._test_normalised_obs(wrapped_env, obs, w_obs)
                for _ in range(self.num_steps):
                    key, _key = jrandom.split(key)
                    action = env.action_space().sample(_key)
                    w_action = wrapped_env.action_space().sample(_key)
                    key, _key = jrandom.split(key)
                    nobs, nenv_state, rew, done, info = env.step(action, env_state, _key)
                    w_nobs, w_nenv_state, w_rew, w_done, w_info = wrapped_env.step(w_action, w_env_state, _key)

                    if normalised and not delta_obs:
                        self._test_normalised_obs(wrapped_env, nobs, w_nobs)

                    if delta_obs:
                        self._test_delta_obs(wrapped_env, obs, nobs, w_nobs, normalised)

                    self._test_rew_fn(rew, action, env_state, nenv_state, w_rew, w_action, w_env_state, w_nenv_state,
                                      env, wrapped_env, _key)

                    obs = copy.deepcopy(nobs)
                    w_obs = copy.deepcopy(w_nobs)
                    env_state = copy.deepcopy(nenv_state)
                    w_env_state = copy.deepcopy(w_nenv_state)

                    if done:
                        break

        except ValueError as e:
            print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test_normal for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    def test_genstep(self, env_name, cont_state, cont_action, normalised, delta_obs):
        try:
            key, _key = jrandom.split(self.key)
            env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                    normalised=False, delta_obs=False, autoreset=False)
            wrapped_env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                            normalised=normalised, delta_obs=delta_obs, autoreset=False)
            # Loop over test episodes
            for _ in range(self.num_episodes):
                obs, env_state = env.reset(_key)
                w_obs, w_env_state = wrapped_env.reset(_key)
                if normalised and not delta_obs:
                    self._test_normalised_obs(wrapped_env, obs, w_obs)
                for _ in range(self.num_steps):
                    key, _key = jrandom.split(key)
                    action = env.action_space().sample(_key)
                    w_action = wrapped_env.action_space().sample(_key)

                    with jax.disable_jit():
                        key, _key = jrandom.split(key)
                        nobs, nenv_state, rew, done, info = env.step(action, env_state, _key)
                        if normalised and not delta_obs:
                            w_nobs, w_nenv_state, w_rew, w_done, w_info = wrapped_env.generative_step(w_action,
                                                                                                     wrapped_env.normalise_obs(obs),
                                                                                                     _key)
                        elif normalised and delta_obs:
                            w_nobs, w_nenv_state, w_rew, w_done, w_info = wrapped_env.generative_step(w_action,
                                                                                                     wrapped_env.test_normalise_obs(obs),
                                                                                                     _key)
                        else:
                            w_nobs, w_nenv_state, w_rew, w_done, w_info = wrapped_env.generative_step(w_action, obs, _key)

                    if normalised and not delta_obs:
                        self._test_normalised_obs(wrapped_env, nobs, w_nobs)

                    if delta_obs:
                        self._test_delta_obs(wrapped_env, obs, nobs, w_nobs, normalised)

                    self._test_rew_fn(rew, action, env_state, nenv_state, w_rew, w_action, w_env_state, w_nenv_state,
                                      env, wrapped_env, _key)


                    obs = copy.deepcopy(nobs)
                    w_obs = copy.deepcopy(w_nobs)
                    env_state = copy.deepcopy(nenv_state)
                    w_env_state = copy.deepcopy(w_nenv_state)

                    if done:
                        break

        except ValueError as e:
            print(
                f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(
                f"Unexpected error during test_genstep for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    def test_autoreset(self, env_name, cont_state, cont_action, normalised, delta_obs):
        try:
            key, _key = jrandom.split(self.key)
            env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                    normalised=False, delta_obs=False, autoreset=False)
            wrapped_env = project_name.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                            normalised=normalised, delta_obs=delta_obs, autoreset=True)
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
                self._test_normalised_obs(wrapped_env, obs, w_obs)

            if delta_obs:
                d_obs = obs[w_dones == 0]
                d_nobs = nobs[w_dones == 0]
                d_w_nobs = w_nobs[w_dones == 0]
                # TODO a dodgy fix as delta obs only works with step but the obs at reset will be wrong
                self._test_delta_obs(wrapped_env, d_obs, d_nobs, d_w_nobs, normalised)

            if not cont_state:
                chex.assert_trees_all_equal(w_obs, obs)

            if normalised and cont_action:
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
                f"Unexpected error during test_autoreset for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
