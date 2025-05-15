import jax
import jax.random as jrandom
import jax.numpy as jnp

import pytest

from project_name.wrappers import NormalisedWrapper
import project_name


class TestWrapper:
    def setup_method(self):
        """Set up common test resources."""
        self.num_steps = 1000
        self.key = jrandom.PRNGKey(42)

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

    @staticmethod
    def _test_obs(wrapped_env, obs):
        unnorm_obs = wrapped_env.unnormalise_obs(obs)
        renorm_obs = wrapped_env.normalise_obs(unnorm_obs)
        assert jnp.allclose(obs, renorm_obs), f"Original obs {obs} not close to renormalised obs {renorm_obs}"

    @staticmethod
    def _test_rew_fn(gt_rew, old_obs, action, obs, wrapped_env, key):
        x = jnp.concatenate([old_obs, action])
        y = obs
        norm_rew = wrapped_env.reward_function(x, y, key)
        assert jnp.allclose(gt_rew, norm_rew), f"gt_rew: {gt_rew}, norm_rew: {norm_rew}"

    @staticmethod
    def _test_update_function(start_obs, action, delta_obs, next_obs, update_fn, env, env_params):
        x = jnp.concatenate([start_obs, action], axis=-1)
        updated_next_obs = update_fn(x, delta_obs, env, env_params)
        assert jnp.allclose(next_obs,
                            updated_next_obs), f"Next obs: {next_obs} and updated next obs: {updated_next_obs}"

    @pytest.mark.parametrize("env_name", ["LogisticMap-v0",
                                          ])
    def test_normalise(self, env_name):
        """Test wrapper normalises okay"""
        key, _key = jrandom.split(self.key)
        env = project_name.make(env_name)
        wrapped_env = NormalisedWrapper(env)
        # regular_update_fn = utils.update_obs_fn
        # wrapped_reward = make_normalised_reward_function(wrapped_env, pendulum_reward)
        # teleport_update_fn = utils.update_obs_fn_teleport
        key, _key = jrandom.split(key)
        obs, env_state = wrapped_env.reset(_key)
        self._test_obs(wrapped_env, obs)
        done = False
        total_rew = 0
        observations = []
        next_observations = []
        rewards = []
        actions = []
        teleport_deltas = []
        for _ in range(self.timesteps):
            old_obs = obs
            observations.append(old_obs)
            key, _key = jrandom.split(key)
            action = wrapped_env.action_space().sample(_key)
            actions.append(action)
            key, _key = jrandom.split(key)
            obs, env_state, rew, done, info = wrapped_env.step(action, env_state, _key)
            next_observations.append(obs)
            total_rew += rew
            standard_delta_obs = obs - old_obs
            teleport_deltas.append(info["delta_obs"])
            # test_update_function(old_obs, action, standard_delta_obs, obs, regular_update_fn, wrapped_env,
            #                      env_params)
            # test_update_function(old_obs, action, info["delta_obs"], obs, teleport_update_fn, wrapped_env,
            #                      env_params)
            rewards.append(rew)
            self._test_obs(wrapped_env, obs)
            key, _key = jrandom.split(key)
            self._test_rew_fn(rew, old_obs, action, obs, wrapped_env, _key)
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
