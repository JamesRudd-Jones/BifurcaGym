import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
import chex
import bifurcagym
import time
import itertools


jax.config.update("jax_enable_x64", True)


env_names = [
             # "Acrobot-v0",
             # "CartPole-v0",
             # "NCartPole-v0",
             # "NPendulum-v0",
             "Pendubot-v0",
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
             # "TentMap-v0",
             # "TinkerbellMap-v0",
             # "FluidicPinball-v0",
             # "BoatInCurrent-v0",
             ]
cont_state = [True, False]
cont_action = [True, False]
# cont_state = [True]
# cont_action = [True]

all_combinations = list(itertools.product(env_names, cont_state, cont_action))

@pytest.mark.parametrize("env_name, cont_state, cont_action",
                         all_combinations)

class TestEnv:
    def setup_method(self, env_name):
        self.num_steps = 100#0
        self.num_episodes = 1#0#0
        self.key = jrandom.key(42)
        self.error = 1e-4


    def test_step(self, env_name, cont_state, cont_action):
        """Test env doesn't have any errors"""
        try:
            key, _key = jrandom.split(self.key)
            env, env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action)

            # Loop over test episodes
            for _ in range(self.num_episodes):
                key, _key = jrandom.split(key)
                obs, state = env.reset(env_params, _key)
                # Loop over test episode steps
                for _ in range(self.num_steps):
                    key, _key = jrandom.split(key)
                    action = env.action_space(env_params).sample(_key)
                    key, _key = jrandom.split(key)
                    obs, delta_obs, state, reward, done_jax, _ = env.step(action, state, env_params, _key)
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
            env, env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action)

            # Loop over test episodes
            for _ in range(self.num_episodes):
                key, _key = jrandom.split(key)
                obs, state = env.reset(env_params, _key)
                # Loop over test episode steps
                for _ in range(self.num_steps):
                    key, _key = jrandom.split(key)
                    action = env.action_space(env_params).sample(_key)

                    key, _key = jrandom.split(key)
                    gen_step_obs, gen_step_delta_obs, gen_step_state, gen_step_reward, gen_step_done, _ = env.generative_step(action,
                                                                                                          obs,
                                                                                                                              env_params,
                                                                                                          _key)
                    step_obs, step_delta_obs, state, step_reward, step_done, _ = env.step(action, state, env_params, _key)

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

                    obs = step_obs

        except ValueError as e:
            print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test_generative_step for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    def test_speed_scan(self, env_name, cont_state, cont_action):
        """Measure the speed of the environment using jax.lax.scan."""
        try:
            key, _ = jrandom.split(self.key)
            env, env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action)

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

    def test_x64_or_x32(self, env_name, cont_state, cont_action):
        try:
            key, _key = jrandom.split(self.key)
            env, env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action)

            obs, state = env.reset(env_params, _key)
            action = env.action_space(env_params).sample(key)
            next_obs, delta_obs, next_state, reward, done, info = env.step(action, state, env_params, key)

            expected_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

            if env.requires_float64:
                expected_float = jnp.float64
                assert obs.dtype == expected_float, f"Expected obs dtype {expected_float}, got {obs.dtype}"
                assert reward.dtype == expected_float, f"Expected reward dtype {expected_float}, got {reward.dtype}"

                leaves = jax.tree_util.tree_leaves(state)
                for i, leaf in enumerate(leaves):
                    # We only check floats, as things like 'time' or 'done' might be ints or bools
                    if jnp.issubdtype(leaf.dtype, jnp.floating):
                        assert leaf.dtype == expected_float, \
                            f"State leaf {i} has incorrect dtype: {leaf.dtype}. Expected {expected_float}."
            else:
                print(f"Environment: {env_name} does not require float64.")



        except ValueError as e:
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test_x64_or_x32: {e}")

