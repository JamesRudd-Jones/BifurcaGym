import jax
import jax.random as jrandom
import jax.numpy as jnp
import pytest
import chex
import bifurcagym
import itertools
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)


env_names = [
             # "Acrobot-v0",
             # "CartPole-v0",
             # "NCartPole-v0",
             # "NPendulum-v0",
             # "Pendubot-v0",
             # "Pendulum-v0",
             # "WetChicken-v0",
             # "ABCFlow-v0",
             # "BickleyJetFlow-v0",
             # "Chua-v0",
             # "DoubleGyreFlow-v0",
             # "KS-v0",
             # "Lorenz63-v0",
             # "QuadrupleGyreFlow-v0",
             # "Rossler-v0",
             # "HenonMap-v0",
             # "IkedaMap-v0",
             # "LogisticMap-v0",
             "TentMap-v0",
             # "TinkerbellMap-v0",
             # "FluidicPinball-v0",
             # "BoatInCurrent-v0",
             ]
cont_state = [True, False]
cont_action = [True, False]
normalised = [True, False]
# cont_state = [True]
# cont_action = [True]
# normalised = [True]

all_combinations = list(itertools.product(env_names,
                                          cont_state,
                                          cont_action,
                                          normalised,
                                          ))

@pytest.mark.parametrize("env_name, "
                         "cont_state, "
                         "cont_action, "
                         "normalised, ",
                         all_combinations)


class TestWrapper:
    def setup_method(self, env_name):
        self.num_steps = 100#0
        self.num_episodes = 10#0
        self.key = jrandom.key(42)
        self.error = 1e-4

    def _test_normalised_obs(self, wrapped_env, obs, w_obs, params):
        # unnormalises obs
        unnorm_obs = wrapped_env.unnormalise_obs(w_obs, params)
        # then renormalises these unormalised obs
        renorm_obs = wrapped_env.normalise_obs(unnorm_obs, params)
        # this test compares the original normalised obs with an unnormalisation and then renormalisation process
        # why?
        chex.assert_trees_all_close(w_obs, renorm_obs, atol=self.error)
        # # this test takes the unnormalised obs and compares them to the non normalised environment
        chex.assert_trees_all_close(unnorm_obs, obs, atol=self.error)

    def _test_delta_obs(self, wrapped_env, obs, nobs, delta_obs, w_obs, w_nobs, w_delta_obs, normalised, params):
        # check delta_obs makes sense
        chex.assert_trees_all_close(nobs, obs + delta_obs, atol=self.error)
        chex.assert_trees_all_close(w_nobs, w_obs + w_delta_obs, atol=self.error)

        if normalised:
            # this test calculates the delta obs by unnormalising the data first and then adding together
            unnorm_w_nobs = wrapped_env.unnormalise_obs(w_obs, params) + wrapped_env.unnormalise_delta_obs(w_delta_obs, params)
            chex.assert_trees_all_close(nobs, unnorm_w_nobs, atol=self.error)

            # this test instead checks the non normalised obs plus the unnormalised delta obs
            unnorm_w_nobs = obs + wrapped_env.unnormalise_delta_obs(w_delta_obs, params)
            chex.assert_trees_all_close(nobs, unnorm_w_nobs, atol=self.error)

    def _test_rew_fn(self, reward_t, action_t, state_t, state_tp1, w_reward_t, w_action_t, w_state_t, w_state_tp1,
                     env, wrapped_env, key, normalised,
                     params, w_params):
        reward, _ = env.reward_and_done_function(action_t, state_t, state_tp1, params, key)
        w_reward, _ = wrapped_env.reward_and_done_function(w_action_t, w_state_t, w_state_tp1, w_params, key)
        chex.assert_trees_all_close(reward, reward_t, atol=self.error)
        chex.assert_trees_all_close(w_reward, w_reward_t, atol=self.error)
        chex.assert_trees_all_close(reward_t, w_reward_t, atol=self.error)

    def _test_apply_delta_obs(self, env, obs, delta_obs, nobs, params):
        chex.assert_trees_all_close(nobs, env.apply_delta_obs(obs, delta_obs, params), atol=self.error)

    # def test_normalised(self, env_name, cont_state, cont_action, normalised):
    #     try:
    #         key, _key = jrandom.split(self.key)
    #         env, env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action,
    #                                           normalised=False, autoreset=False, metrics=False)
    #         wrapped_env, w_env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action,
    #                                                     normalised=normalised, autoreset=False, metrics=False)
    #         # TODO wrapping has no effect on the params but we highlight the difference here anyway
    #
    #         # Loop over test episodes
    #         for _ in range(self.num_episodes):
    #             obs, env_state = env.reset(env_params, _key)
    #             w_obs, w_env_state = wrapped_env.reset(w_env_params, _key)
    #             if normalised:
    #                 self._test_normalised_obs(wrapped_env, obs, w_obs, w_env_params)
    #             for _ in range(self.num_steps):
    #                 key, _key = jrandom.split(key)
    #                 action = env.action_space(env_params).sample(_key)
    #                 w_action = wrapped_env.action_space(w_env_params).sample(_key)
    #                 key, _key = jrandom.split(key)
    #                 nobs, delta_obs, nenv_state, rew, done, info = env.step(action, env_state, env_params, _key)
    #                 w_nobs, w_delta_obs, w_nenv_state, w_rew, w_done, w_info = wrapped_env.step(w_action,
    #                                                                                             w_env_state,
    #                                                                                             w_env_params,
    #                                                                                             _key)
    #
    #                 if normalised:
    #                     self._test_normalised_obs(wrapped_env, nobs, w_nobs, w_env_params)
    #
    #                 self._test_delta_obs(wrapped_env, obs, nobs, delta_obs, w_obs, w_nobs, w_delta_obs, normalised, w_env_params)
    #
    #                 self._test_rew_fn(rew, action, env_state, nenv_state, w_rew, w_action, w_env_state, w_nenv_state,
    #                                   env, wrapped_env, _key, normalised, env_params, w_env_params)
    #
    #                 self._test_apply_delta_obs(env, obs, delta_obs, nobs, env_params)
    #                 self._test_apply_delta_obs(wrapped_env, w_obs, w_delta_obs, w_nobs, w_env_params)
    #
    #                 obs = nobs
    #                 w_obs = w_nobs
    #                 env_state = nenv_state
    #                 w_env_state = w_nenv_state
    #
    #                 if done:
    #                     break
    #
    #     except ValueError as e:
    #         print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
    #         pytest.skip(f"Skipping test due to expected ValueError: {e}")
    #     except Exception as e:
    #         pytest.fail(f"Unexpected error during test_normal for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
    #
    # def test_genstep(self, env_name, cont_state, cont_action, normalised):
    #     try:
    #         key, _key = jrandom.split(self.key)
    #         env, env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action,
    #                                           normalised=False, autoreset=False)
    #         wrapped_env, w_env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action,
    #                                                     normalised=normalised, autoreset=False)
    #         # Loop over test episodes
    #         for _ in range(self.num_episodes):
    #             obs, env_state = env.reset(env_params, _key)
    #             w_obs, w_env_state = wrapped_env.reset(w_env_params, _key)
    #             if normalised:
    #                 self._test_normalised_obs(wrapped_env, obs, w_obs, w_env_params)
    #             for _ in range(self.num_steps):
    #                 key, _key = jrandom.split(key)
    #                 action = env.action_space(env_params).sample(_key)
    #                 w_action = wrapped_env.action_space(w_env_params).sample(_key)
    #
    #                 with jax.disable_jit():
    #                     key, _key = jrandom.split(key)
    #                     nobs, delta_obs, nenv_state, rew, done, info = env.step(action, env_state, env_params, _key)
    #                     if normalised and cont_state:
    #                         w_nobs, w_delta_obs, w_nenv_state, w_rew, w_done, w_info = wrapped_env.generative_step(w_action,
    #                                                                                                                wrapped_env.normalise_obs(obs, w_env_params),
    #                                                                                                                w_env_params,
    #                                                                                                                _key)
    #                         # Generally if we normalise then the obs that get fed in are also normalised I think
    #                         # Equivalent to the above is feeding in w_obs as this should be the same as normalised(obs)
    #                     elif not cont_state:
    #                         w_nobs, w_delta_obs, w_nenv_state, w_rew, w_done, w_info = wrapped_env.generative_step(w_action,
    #                                                                                                                # env_state.x,
    #                                                                                                                wrapped_env.get_obs(w_env_state, _key),
    #                                                                                                                w_env_params,
    #                                                                                                                _key)
    #                         # TODO a dodgy fix for now due to the discretisation thing with get_obs
    #                     else:
    #                         w_nobs, w_delta_obs, w_nenv_state, w_rew, w_done, w_info = wrapped_env.generative_step(w_action,
    #                                                                                                                obs,
    #                                                                                                                w_env_params,
    #                                                                                                                _key)
    #
    #                 if normalised:
    #                     self._test_normalised_obs(wrapped_env, nobs, w_nobs, w_env_params)
    #
    #                 self._test_delta_obs(wrapped_env, obs, nobs, delta_obs, w_obs, w_nobs, w_delta_obs, normalised,
    #                                      w_env_params)
    #
    #                 self._test_rew_fn(rew, action, env_state, nenv_state, w_rew, w_action, w_env_state, w_nenv_state,
    #                                   env, wrapped_env, _key, normalised, env_params, w_env_params)
    #
    #
    #                 obs = nobs
    #                 w_obs = w_nobs
    #                 env_state = nenv_state
    #                 w_env_state = w_nenv_state
    #
    #                 if done:
    #                     break
    #
    #     except ValueError as e:
    #         print(
    #             f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
    #         pytest.skip(f"Skipping test due to expected ValueError: {e}")
    #     except Exception as e:
    #         pytest.fail(
    #             f"Unexpected error during test_genstep for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    # def test_autoreset(self, env_name, cont_state, cont_action, normalised):
    #     try:
    #         key, _key = jrandom.split(self.key)
    #         env, env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action,
    #                                           normalised=False, autoreset=False)
    #         wrapped_env, w_env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action,
    #                                                     normalised=normalised, autoreset=True)
    #         observations = []
    #         delta_observations = []
    #         nobservations = []
    #         rewards = []
    #         actions = []
    #         dones = []
    #
    #         # Loop over test episodes
    #         obs, env_state = env.reset(env_params, _key)
    #         for _ in range(self.num_episodes):
    #             for _ in range(self.num_steps):
    #                 key, _key = jrandom.split(key)
    #                 action = env.action_space(env_params).sample(_key)
    #                 key, _key = jrandom.split(key)
    #                 nobs, delta_obs, env_state, rew, done, info = env.step(action, env_state, env_params, _key)
    #                 # _key is 3122727659, 270479714
    #                 observations.append(obs)
    #                 delta_observations.append(delta_obs)
    #                 nobservations.append(nobs)
    #                 actions.append(action)
    #                 rewards.append(rew)
    #                 dones.append(done)
    #
    #                 obs = nobs
    #
    #                 if done:
    #                     _, reset_key = jrandom.split(_key)
    #                     # reset_key is 3320941097, 3018999918
    #                     obs, env_state = env.reset(env_params, reset_key)
    #                     break
    #
    #         obs = jnp.array(observations)
    #         nobs = jnp.array(nobservations)
    #         delta_obs = jnp.array(delta_observations)
    #         actions = jnp.array(actions)
    #         rewards = jnp.array(rewards)
    #         dones = jnp.array(dones)
    #
    #         def scan_step(carry, _):
    #             state, obs, params, key = carry
    #             key, _key = jrandom.split(key)
    #             action = wrapped_env.action_space(params).sample(_key)
    #             key, _key = jrandom.split(key)
    #             nobs, delta_obs, next_state, reward, done, info = wrapped_env.step(action, state, params,_key)
    #             return (next_state, nobs, params, key), (obs, nobs, delta_obs, action, reward, done)
    #
    #         key, _key = jrandom.split(self.key)
    #         init_w_obs, w_env_state = wrapped_env.reset(w_env_params, _key)
    #         with jax.disable_jit(disable=False):
    #             (final_state, _, _, _), (w_obs, w_nobs, w_delta_obs, w_actions, w_rewards,
    #                                      w_dones) = jax.lax.scan(scan_step, (w_env_state, init_w_obs, w_env_params, key), None, self.num_steps * self.num_episodes)
    #
    #         obs_length = len(obs)
    #         w_obs = w_obs[:obs_length]
    #         w_nobs = w_nobs[:obs_length]
    #         w_delta_obs = w_delta_obs[:obs_length]
    #         w_actions = w_actions[:obs_length]
    #         w_rewards = w_rewards[:obs_length]
    #         w_dones = w_dones[:obs_length]
    #
    #         if normalised:
    #             self._test_normalised_obs(wrapped_env, obs, w_obs, w_env_params)
    #         else:
    #             chex.assert_trees_all_close(w_obs, obs, atol=self.error)
    #
    #         d_obs = obs[w_dones == 0]
    #         d_nobs = nobs[w_dones == 0]
    #         d_delta_obs = delta_obs[w_dones == 0]
    #         d_w_obs = w_obs[w_dones == 0]
    #         d_w_nobs = w_nobs[w_dones == 0]
    #         d_w_delta_obs = w_delta_obs[w_dones == 0]
    #         # TODO a dodgy fix as delta obs only works with step but the obs at reset will be wrong
    #         self._test_delta_obs(wrapped_env, d_obs, d_nobs, d_delta_obs, d_w_obs, d_w_nobs, d_w_delta_obs, normalised,
    #                              w_env_params)
    #
    #         if not cont_state:
    #             chex.assert_trees_all_equal(w_obs, obs)
    #
    #         if normalised and cont_action:
    #             chex.assert_trees_all_close(wrapped_env.unnormalise_action(w_actions, w_env_params), actions, atol=self.error)
    #         else:
    #             if cont_action:
    #                 chex.assert_trees_all_close(w_actions, actions, atol=self.error)
    #             else:
    #                 chex.assert_trees_all_equal(w_actions, actions)
    #
    #         chex.assert_trees_all_close(w_rewards, rewards, atol=self.error)
    #         chex.assert_trees_all_equal(w_dones, dones)
    #
    #     except ValueError as e:
    #         print(
    #             f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
    #         pytest.skip(f"Skipping test due to expected ValueError: {e}")
    #     except Exception as e:
    #         pytest.fail(
    #             f"Unexpected error during test_autoreset for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    def test_autoreset_fixed(self, env_name, cont_state, cont_action, normalised):
        try:
            env, env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                              normalised=False, autoreset=True)
            wrapped_env, w_env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                                        normalised=normalised, autoreset=True)

            total_steps = self.num_steps * self.num_episodes

            key, action_key, reset_key, step_key = jrandom.split(self.key, 4)

            # Pre-sample actions using the UNWRAPPED environment space
            action_keys = jrandom.split(action_key, total_steps)
            raw_actions = jax.vmap(env.action_space(env_params).sample)(action_keys)

            # Translate actions for the wrapper
            if normalised and cont_action:
                w_actions = jax.vmap(lambda a: wrapped_env.normalise_action(a, w_env_params))(raw_actions)
            else:
                w_actions = raw_actions

            def unwrapped_scan_step(carry, action):
                state, params, k = carry
                k, sk = jrandom.split(k)
                nobs, delta_obs, next_state, reward, done, info = env.step(action, state, params, sk)
                return (next_state, params, k), (nobs, delta_obs, action, reward, done)

            def wrapped_scan_step(carry, action):
                state, params, k = carry
                k, sk = jrandom.split(k)
                nobs, delta_obs, next_state, reward, done, info = wrapped_env.step(action, state, params, sk)
                return (next_state, params, k), (nobs, delta_obs, action, reward, done)

            init_obs, env_state = env.reset(env_params, reset_key)
            init_w_obs, w_env_state = wrapped_env.reset(w_env_params, reset_key)

            _, (nobs, delta_obs, actions, rewards, dones) = jax.lax.scan(
                unwrapped_scan_step, (env_state, env_params, step_key), raw_actions
            )
            _, (w_nobs, w_delta_obs, w_actions_out, w_rewards, w_dones) = jax.lax.scan(
                wrapped_scan_step, (w_env_state, w_env_params, step_key), w_actions
            )

            # Reconstruct the observation history (obs_t instead of obs_tp1)
            # This makes the array length match the scan output
            obs_history = jnp.concatenate([init_obs[None, ...], nobs[:-1]])
            w_obs_history = jnp.concatenate([init_w_obs[None, ...], w_nobs[:-1]])

            # Observation Parity
            if normalised:
                vmap_unnorm = jax.vmap(lambda o: wrapped_env.unnormalise_obs(o, w_env_params))
                unnorm_w_obs_history = vmap_unnorm(w_obs_history)
                # diff = jnp.abs(obs_history - unnorm_w_obs_history)
                # plt.plot(diff[:, 0], label="Dim 1")
                # plt.plot(diff[:, 1], label="Dim 2")
                # plt.plot(diff[:, 2], label="Dim 3")
                # plt.plot(diff[:, 3], label="Dim 4")
                # plt.plot(diff[:, 4], label="Dim 5")
                # plt.plot(diff[:, 5], label="Dim 6")
                # plt.legend()
                # # plt.xlim(500)
                # plt.ylim(0, 0.2)
                # plt.show()
                chex.assert_trees_all_close(obs_history, unnorm_w_obs_history, atol=self.error)
            elif cont_state:
                chex.assert_trees_all_close(w_obs_history, obs_history, atol=self.error)
            else:
                chex.assert_trees_all_equal(w_obs_history, obs_history)

            # Action Parity
            if normalised and cont_action:
                vmap_unnorm_act = jax.vmap(lambda a: wrapped_env.unnormalise_action(a, w_env_params))
                chex.assert_trees_all_close(vmap_unnorm_act(w_actions_out), actions, atol=self.error)
            elif cont_action:
                chex.assert_trees_all_close(w_actions_out, actions, atol=self.error)
            else:
                chex.assert_trees_all_equal(w_actions_out, actions)

            # Reward and Done State Parity
            chex.assert_trees_all_close(w_rewards, rewards, atol=self.error)
            chex.assert_trees_all_equal(w_dones, dones)

            # Delta Observation Parity (Excluding boundaries)
            # Delta obs is mathematically invalid across an autoreset boundary.
            # We filter for steps where the previous state was NOT done.
            valid_mask = dones == 0
            d_obs = obs_history[valid_mask]
            d_nobs = nobs[valid_mask]
            d_delta = delta_obs[valid_mask]
            d_w_obs = w_obs_history[valid_mask]
            d_w_nobs = w_nobs[valid_mask]
            d_w_delta = w_delta_obs[valid_mask]

            if len(d_obs) > 0:
                vmap_unnorm_delta = jax.vmap(lambda do: wrapped_env.unnormalise_delta_obs(do, w_env_params))
                vmap_unnorm_obs = jax.vmap(lambda o: wrapped_env.unnormalise_obs(o, w_env_params))

                chex.assert_trees_all_close(d_nobs, d_obs + d_delta, atol=self.error)
                chex.assert_trees_all_close(d_w_nobs, d_w_obs + d_w_delta, atol=self.error)

                if normalised:
                    unnorm_w_nobs = vmap_unnorm_obs(d_w_obs) + vmap_unnorm_delta(d_w_delta)
                    chex.assert_trees_all_close(d_nobs, unnorm_w_nobs, atol=self.error)

        except ValueError as e:
            print(
                f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(
                f"Unexpected error during test_autoreset_fixed for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    def test_autoreset_boundary(self, env_name, cont_state, cont_action, normalised):
        try:
            wrapped_env, w_env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action,
                                                        normalised=normalised, autoreset=True)

            def cond_fn(val):
                return jnp.logical_and(jnp.logical_not(val[3]), val[4] < 20000)

            def step_fn(val):
                obs, state, key, _, steps = val
                key, action_key, step_key = jrandom.split(key, 3)
                action = wrapped_env.action_space(w_env_params).sample(action_key)
                next_obs, delta_obs, next_state, rew, done, info = wrapped_env.step(action, state, w_env_params, step_key)

                return (next_obs, next_state, key, done, steps + 1)

            key, reset_key = jrandom.split(self.key)
            init_obs, init_state = wrapped_env.reset(w_env_params, reset_key)

            final_obs, final_state, _, hit_done, steps_taken = jax.lax.while_loop(
                cond_fn, step_fn, (init_obs, init_state, key, False, 0)
            )

            assert hit_done, f"Environment {env_name} did not reach a done state."

            chex.assert_tree_all_finite(final_obs)

            if normalised:
                low_bound = wrapped_env.observation_space(w_env_params).low
                high_bound = wrapped_env.observation_space(w_env_params).high

                assert jnp.all(final_obs >= low_bound - self.error), "Autoreset observation is below normalised bounds"
                assert jnp.all(final_obs <= high_bound + self.error), "Autoreset observation is above normalised bounds"

                unnorm_final_obs = wrapped_env.unnormalise_obs(final_obs, w_env_params)
                chex.assert_tree_all_finite(unnorm_final_obs)


        except ValueError as e:
            print(
                f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")

        except Exception as e:
            pytest.fail(
                f"Unexpected error during test_boundary for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
