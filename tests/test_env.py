import jax
import jax.numpy as jnp
import jax.random as jrandom

import pytest

import chex

import bifurcagym
# from gymnax.utils import state_translate, test_helpers

from copy import deepcopy
import time
import itertools


env_names = [
             # "Acrobot-v0",
             # "CartPole-v0",
             # "NCartPole-v0",
             # "Pendubot-v0",
             "Pendulum-v0",
             # "WetChicken-v0",
             #  "KS-v0",
             # "LogisticMap-v0",
             # "TentMap-v0",
             ]
cont_state = [True, False]
cont_action = [True, False]

all_combinations = list(itertools.product(env_names, cont_state, cont_action))

@pytest.mark.parametrize("env_name, cont_state, cont_action",
                         all_combinations)

class TestEnv:
    def setup_method(self):
        """Set up common test resources."""
        self.num_steps = 100#0
        self.num_episodes = 10#0
        self.key = jrandom.PRNGKey(42)
        self.error = 1e-4


    def test_step(self, env_name, cont_state, cont_action):
        """Test env doesn't have any errors"""
        try:
            key, _key = jrandom.split(self.key)
            env = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action)

            # Loop over test episodes
            for _ in range(self.num_episodes):
                key, _key = jrandom.split(key)
                obs, state = env.reset(_key)
                # Loop over test episode steps
                for _ in range(self.num_steps):
                    key, _key = jrandom.split(key)
                    action = env.action_space().sample(_key)
                    key, _key = jrandom.split(key)
                    obs, delta_obs, state, reward, done_jax, _ = env.step(action, state, _key)
                    # print(obs)

        except ValueError as e:
            print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test_step for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    def test_generative_step(self, env_name, cont_state, cont_action):
        """Ensure generative step matches the normal step"""
        try:
            key, _key = jrandom.split(self.key)
            env = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action)

            # Loop over test episodes
            for _ in range(self.num_episodes):
                key, _key = jrandom.split(key)
                obs, state = env.reset(_key)
                # Loop over test episode steps
                for _ in range(self.num_steps):
                    key, _key = jrandom.split(key)
                    action = env.action_space().sample(_key)

                    key, _key = jrandom.split(key)
                    gen_step_obs, gen_step_delta_obs, gen_step_state, gen_step_reward, gen_step_done, _ = env.generative_step(action,
                                                                                                          obs,
                                                                                                          _key)
                    step_obs, step_delta_obs, state, step_reward, step_done, _ = env.step(action, state, _key)

                    if not cont_state:
                        chex.assert_trees_all_equal(step_obs, gen_step_obs)
                        chex.assert_trees_all_equal(step_delta_obs, gen_step_delta_obs)
                    else:
                        chex.assert_trees_all_close(step_obs, gen_step_obs, atol=self.error)
                        chex.assert_trees_all_close(step_delta_obs, gen_step_delta_obs, atol=self.error)
                    gen_step_state = gen_step_state.replace(time=state.time)
                    chex.assert_trees_all_close(state, gen_step_state, atol=self.error)
                    chex.assert_trees_all_close(step_reward, gen_step_reward, atol=self.error)
                    chex.assert_trees_all_equal(step_done, gen_step_done)

                    obs = deepcopy(step_obs)

        except ValueError as e:
            print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test_generative_step for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    def test_speed_scan(self, env_name, cont_state, cont_action):
        """Measure the speed of the environment using jax.lax.scan."""
        try:
            key, _ = jrandom.split(self.key)
            env = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action)

            def scan_step(carry, _):
                state, key = carry
                key, _key = jrandom.split(key)
                action = env.action_space().sample(_key)
                key, _key = jrandom.split(key)
                obs, delta_obs, next_state, reward, done, info = env.step(action, state, _key)
                return (next_state, key), None

            start_time = time.time()

            for _ in range(self.num_episodes):
                key, _key = jrandom.split(key)
                _, states = env.reset(_key)
                (final_state, _), _ = jax.lax.scan(scan_step, (states, key), None, self.num_steps)

            end_time = time.time()
            elapsed_time = end_time - start_time
            steps_per_second = (self.num_steps * self.num_episodes) / elapsed_time

            print(f"\n--- Speed Test (Scan) for {env_name} ---")
            print(f"Ran {self.num_steps * self.num_episodes} steps in {elapsed_time:.4f} seconds.")
            print(f"Speed: {steps_per_second:.2f} steps/second")

        except ValueError as e:
            print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test_speed_scan for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

