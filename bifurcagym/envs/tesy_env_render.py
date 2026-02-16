import jax
import jax.numpy as jnp
import jax.random as jrandom

import chex

import bifurcagym
# from gymnax.utils import state_translate, test_helpers

from copy import deepcopy
import time
import itertools

import matplotlib.pyplot as plt


num_steps = 50#0#0
num_episodes = 1#00
key = jrandom.key(42)
error = 1e-4
env_name = "KS-v0"

def test_step(env_name, cont_state, cont_action, normalised, autoreset, metrics, key):
    key, _ = jrandom.split(key)
    env = bifurcagym.make(env_name,
                          cont_state=cont_state,
                          cont_action=cont_action,
                          normalised=normalised,
                          autoreset=autoreset,
                          metrics=metrics)

    def scan_step(carry, _):
        state, key = carry
        key, _key = jrandom.split(key)
        action = env.action_space().sample(_key)
        key, _key = jrandom.split(key)
        obs, delta_obs, next_state, reward, done, info = env.step(action, state, _key)
        return (next_state, key), (state, obs, delta_obs)

    for _ in range(num_episodes):
        key, _key = jrandom.split(key)
        init_obs, states = env.reset(_key)
        (final_state, _), (state, obs, delta_obs) = jax.lax.scan(scan_step, (states, key), None, num_steps)

    total_state = jax.tree.map(lambda x, y: jnp.concatenate((x, jnp.expand_dims(y, axis=0))), state, final_state)

    # def reconstruct_observations(initial_obs, delta_obs_trajectory):
    #     def scan_fn(carry, delta_obs):
    #         next_obs = carry + delta_obs
    #         return next_obs, next_obs
    #
    #     # Initialize the carry with the initial observation
    #     initial_carry = initial_obs
    #
    #     # Perform the scan
    #     _, reconstructed_obs_tail = jax.lax.scan(scan_fn, initial_carry, delta_obs_trajectory)
    #
    #     # Prepend the initial observation to the result
    #     reconstructed_obs = jnp.concatenate([initial_obs[None, ...], reconstructed_obs_tail], axis=0)
    #     return reconstructed_obs
    #
    # total_obs = reconstruct_observations(init_obs, obs)

    env.render_traj(total_state)
    plt.scatter(total_obs[:, 0], total_obs[:, 1])
    plt.scatter(total_obs[0, 0], total_obs[0, 1], color='r')
    plt.xlabel("$\\theta$")
    plt.ylabel("$\\dot{\\theta}$")
    plt.show()

test_step(env_name,
          cont_state=True,
          cont_action=False,
          normalised=False,
          autoreset=True,
          metrics=False,
          key=key)

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
