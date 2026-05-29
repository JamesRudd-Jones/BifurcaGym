import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
import chex
import bifurcagym
from copy import deepcopy
import itertools


env_names = [
             # "Acrobot-v0",
             # "CartPole-v0",
             "NCartPole-v0",
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
             # "TentMap-v0",
             # "TinkerbellMap-v0",
             # "FluidicPinball-v0",
             # "BoatInCurrent-v0",
             ]
cont_state = [True, False]
cont_action = [True, False]
# cont_state = [True]
# cont_action = [False]

all_combinations = list(itertools.product(env_names, cont_state, cont_action))

@pytest.mark.parametrize("env_name, cont_state, cont_action",
                         all_combinations)

class TestEnv:
    def setup_method(self, env_name):
        self.num_steps = 200#0
        self.num_episodes = 100
        self.key = jrandom.key(42)
        self.error = 1e-4


    def test_render(self, env_name, cont_state, cont_action):
        """Test env doesn't have any errors"""
        try:
            key, _key = jrandom.split(self.key)
            env, env_params = bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action, autoreset=True)

            key, _key = jrandom.split(key)
            obs, env_state = env.reset(env_params, _key)

            def _loop_func(runner_state, unused):
                obs, env_state, env_params, key = runner_state
                key, _key = jrandom.split(key)
                action = env.action_space(env_params).sample(_key)
                # action = jnp.sin(0.5 * env_state.time) * 0.1 * jnp.cos(env_state.time)
                # action = jnp.ones(1,) * env.action_space().high
                # action = jnp.zeros(env.action_space().shape[0])
                key, _key = jrandom.split(key)
                nobs, delta_obs, next_env_state, rew, done, info = env.step(action, env_state, env_params, _key)

                return (nobs, next_env_state, env_params, key), (env_state, env_params)

            _, (traj_state, traj_params) = jax.lax.scan(_loop_func, (obs, env_state, env_params, key), None, self.num_steps)

            env.render_traj(traj_state, traj_params, "./animations/")

        except ValueError as e:
            print(f"Caught expected ValueError for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")
            pytest.skip(f"Skipping test due to expected ValueError: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during test_step for {env_name} with cont_state={cont_state}, cont_action={cont_action}: {e}")

    # TODO do I need to test rendering for generative stuff as well or is it assumed to work if the gen test passes?
