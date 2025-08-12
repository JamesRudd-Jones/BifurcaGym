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
             # "Pendulum-v0",
             # "WetChicken-v0",
             #  "KS-v0",
             # "LogisticMap-v0",
             # "TentMap-v0",
             "BoatInCurrent-v0",
             ]
cont_state = [True]#, False]
cont_action = [True]#, False]

all_combinations = list(itertools.product(env_names, cont_state, cont_action))

@pytest.mark.parametrize("env_name, cont_state, cont_action",
                         all_combinations)

class TestEnv:
    def setup_method(self):
        self.num_steps = 200#0
        self.num_episodes = 100
        self.key = jrandom.key(42)
        self.error = 1e-4


    def test_step(self, env_name, cont_state, cont_action):
        """Test env doesn't have any errors"""
        try:
            key, _key = jrandom.split(self.key)
            env = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action, autoreset=True)

            key, _key = jrandom.split(key)
            obs, env_state = env.reset(_key)

            def _loop_func(runner_state, unused):
                obs, env_state, key = runner_state
                key, _key = jrandom.split(key)
                # action = env.action_space().sample(_key)
                # action = jnp.sin(0.5 * env_state.time) * 0.1 * jnp.cos(env_state.time)
                action = jnp.ones(1,) * env.action_space().high
                key, _key = jrandom.split(key)
                nobs, delta_obs, next_env_state, rew, done, info = env.step(action, env_state, _key)

                return (nobs, next_env_state, key), env_state

            _, traj = jax.lax.scan(_loop_func, (obs, env_state, key), None, self.num_steps)

            env.render_traj(traj)

        except ValueError as e:
            print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test_step for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    # def test_generative_step(self, env_name, cont_state, cont_action):
    #     """Ensure generative step matches the normal step"""
    #     try:
    #         key, _key = jrandom.split(self.key)
    #         env = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action)
    #
    #         # Loop over test episodes
    #         for _ in range(self.num_episodes):
    #             key, _key = jrandom.split(key)
    #             obs, state = env.reset(_key)
    #             # Loop over test episode steps
    #             for _ in range(self.num_steps):
    #                 key, _key = jrandom.split(key)
    #                 action = env.action_space().sample(_key)
    #
    #                 key, _key = jrandom.split(key)
    #                 # TODO a bit dodgy as the state.x may change, not sure how to generalise atm
    #                 gen_step_obs, gen_step_state, gen_step_reward, gen_step_done, _ = env.generative_step(action, state.x, _key)
    #                 step_obs, state, step_reward, step_done, _ = env.step(action, state, _key)
    #
    #                 if obs.dtype == jnp.int32 or obs.dtype == jnp.int64:
    #                     chex.assert_trees_all_equal(step_obs, gen_step_obs)
    #                 else:
    #                     chex.assert_trees_all_close(step_obs, gen_step_obs, atol=self.error)
    #                 gen_step_state = gen_step_state.replace(time=state.time)
    #                 chex.assert_trees_all_close(state, gen_step_state)
    #                 chex.assert_trees_all_close(step_reward, gen_step_reward, atol=self.error)
    #                 chex.assert_trees_all_equal(step_done, gen_step_done)
    #
    #                 obs = deepcopy(step_obs)
    #
    #     except ValueError as e:
    #         print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
    #         pytest.skip(f"Skipping test due to expected ValueError: {e}")
    #     except Exception as e:
    #         pytest.fail(f"Unexpected error during test_generative_step for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
