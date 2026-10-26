import time

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from conftest import env_combinations, make_or_skip, supports_generative_step


@pytest.mark.parametrize("env_name, cont_state, cont_action", env_combinations())
class TestEnv:
    def setup_method(self):
        self.num_steps = 100
        self.num_episodes = 1
        self.key = jrandom.key(42)
        self.error = 1e-4

    def test_step(self, env_name, cont_state, cont_action):
        """Test env doesn't have any errors and outputs stay finite and consistent"""
        key, _key = jrandom.split(self.key)
        env, env_params = make_or_skip(env_name, cont_state, cont_action)

        # Loop over test episodes
        for _ in range(self.num_episodes):
            key, _key = jrandom.split(key)
            obs, state = env.reset(env_params, _key)
            chex.assert_tree_all_finite(obs)
            if cont_state:
                chex.assert_shape(obs, env.observation_space(env_params).shape)
            # Loop over test episode steps
            for _ in range(self.num_steps):
                key, _key = jrandom.split(key)
                action = env.action_space(env_params).sample(_key)
                key, _key = jrandom.split(key)
                nobs, delta_obs, state, reward, done, _ = env.step(action, state, env_params, _key)

                chex.assert_shape(reward, ())
                chex.assert_shape(done, ())
                if cont_state:
                    chex.assert_trees_all_close(env.apply_delta_obs(obs, delta_obs, env_params), nobs,
                                                atol=self.error)

                obs = nobs
                if done:
                    break

    def test_reset_deterministic(self, env_name, cont_state, cont_action):
        """Same key gives the same reset"""
        env, env_params = make_or_skip(env_name, cont_state, cont_action)

        obs_a, state_a = env.reset(env_params, self.key)
        obs_b, state_b = env.reset(env_params, self.key)

        chex.assert_trees_all_equal(obs_a, obs_b)
        chex.assert_trees_all_equal(state_a, state_b)

    def test_reward_function_matches_step(self, env_name, cont_state, cont_action):
        """The standalone reward function reproduces the reward given by step, needed for Model-Based RL"""
        key, _key = jrandom.split(self.key)
        env, env_params = make_or_skip(env_name, cont_state, cont_action)

        _, state = env.reset(env_params, _key)
        for _ in range(10):
            key, _key = jrandom.split(key)
            action = env.action_space(env_params).sample(_key)
            key, _key = jrandom.split(key)
            _, _, nstate, reward, done, _ = env.step(action, state, env_params, _key)

            fn_reward, fn_done = env.reward_and_done_function(action, state, nstate, env_params, _key)
            chex.assert_trees_all_close(fn_reward, reward, atol=self.error)
            chex.assert_trees_all_equal(fn_done, done)

            state = nstate
            if done:
                break

    def test_generative_step(self, env_name, cont_state, cont_action):
        """Ensure generative step matches the normal step"""
        key, _key = jrandom.split(self.key)
        env, env_params = make_or_skip(env_name, cont_state, cont_action)

        if not supports_generative_step(env_name, cont_state):
            key, _key = jrandom.split(key)
            obs, _ = env.reset(env_params, _key)
            action = env.action_space(env_params).sample(_key)
            with pytest.raises(ValueError):
                env.generative_step(action, obs, env_params, _key)
            return

        # Loop over test episodes
        for _ in range(self.num_episodes):
            key, _key = jrandom.split(key)
            obs, state = env.reset(env_params, _key)
            # Loop over test episode steps
            for _ in range(self.num_steps):
                key, _key = jrandom.split(key)
                action = env.action_space(env_params).sample(_key)

                key, _key = jrandom.split(key)
                (gen_step_obs, gen_step_delta_obs, gen_step_state,
                 gen_step_reward, gen_step_done, _) = env.generative_step(action, obs, env_params, _key)
                step_obs, step_delta_obs, state, step_reward, step_done, _ = env.step(action, state, env_params, _key)

                chex.assert_trees_all_close(step_obs, gen_step_obs, atol=self.error)
                chex.assert_trees_all_close(step_delta_obs, gen_step_delta_obs, atol=self.error)
                gen_step_state = gen_step_state.replace(time=state.time)
                chex.assert_trees_all_close(state, gen_step_state, atol=self.error)
                chex.assert_trees_all_close(step_reward, gen_step_reward, atol=self.error)
                chex.assert_trees_all_equal(step_done, gen_step_done)

                obs = step_obs

    def test_speed_scan(self, env_name, cont_state, cont_action):
        """Measure the speed of the environment using jax.lax.scan."""
        key, _ = jrandom.split(self.key)
        env, env_params = make_or_skip(env_name, cont_state, cont_action)

        def scan_step(carry, _):
            state, env_params, key = carry
            key, _key = jrandom.split(key)
            action = env.action_space(env_params).sample(_key)
            key, _key = jrandom.split(key)
            obs, delta_obs, next_state, reward, done, info = env.step(action, state, env_params, _key)
            return (next_state, env_params, key), None

        start_time = time.time()

        for _ in range(self.num_episodes):
            key, _key = jrandom.split(key)
            _, states = env.reset(env_params, _key)
            (final_state, _, _), _ = jax.lax.scan(scan_step, (states, env_params, key), None, self.num_steps)
            jax.block_until_ready(final_state)

        end_time = time.time()
        elapsed_time = end_time - start_time
        steps_per_second = (self.num_steps * self.num_episodes) / elapsed_time

        print(f"\n--- Speed Test (Scan) for {env_name} ---")
        print(f"Ran {self.num_steps * self.num_episodes} steps in {elapsed_time:.4f} seconds.")
        print(f"Speed: {steps_per_second:.2f} steps/second")

    def test_x64_or_x32(self, env_name, cont_state, cont_action):
        key, _key = jrandom.split(self.key)
        env, env_params = make_or_skip(env_name, cont_state, cont_action)

        obs, state = env.reset(env_params, _key)
        action = env.action_space(env_params).sample(key)
        next_obs, delta_obs, next_state, reward, done, info = env.step(action, state, env_params, key)

        if not env.requires_float64:
            pytest.skip(f"Environment: {env_name} does not require float64.")

        expected_float = jnp.float64
        if cont_state:
            assert next_obs.dtype == expected_float, f"Expected obs dtype {expected_float}, got {next_obs.dtype}"
        assert reward.dtype == expected_float, f"Expected reward dtype {expected_float}, got {reward.dtype}"

        leaves = jax.tree_util.tree_leaves(next_state)
        for i, leaf in enumerate(leaves):
            # We only check floats, as things like 'time' or 'done' might be ints or bools
            if jnp.issubdtype(leaf.dtype, jnp.floating):
                assert leaf.dtype == expected_float, \
                    f"State leaf {i} has incorrect dtype: {leaf.dtype}. Expected {expected_float}."
