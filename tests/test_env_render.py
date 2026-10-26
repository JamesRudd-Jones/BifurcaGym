import jax
import jax.random as jrandom
import pytest

from conftest import env_combinations, make_or_skip


@pytest.mark.render
@pytest.mark.parametrize("env_name, cont_state, cont_action", env_combinations())
class TestEnvRender:
    def setup_method(self):
        self.num_steps = 200
        self.key = jrandom.key(42)

    def test_render(self, env_name, cont_state, cont_action, tmp_path):
        """Test rendering a trajectory doesn't have any errors"""
        key, _key = jrandom.split(self.key)
        env, env_params = make_or_skip(env_name, cont_state, cont_action, autoreset=True)

        key, _key = jrandom.split(key)
        obs, env_state = env.reset(env_params, _key)

        def _loop_func(runner_state, unused):
            obs, env_state, env_params, key = runner_state
            key, _key = jrandom.split(key)
            action = env.action_space(env_params).sample(_key)
            key, _key = jrandom.split(key)
            nobs, delta_obs, next_env_state, rew, done, info = env.step(action, env_state, env_params, _key)

            return (nobs, next_env_state, env_params, key), (env_state, env_params)

        _, (traj_state, traj_params) = jax.lax.scan(_loop_func, (obs, env_state, env_params, key), None, self.num_steps)

        env.render_traj(traj_state, traj_params, str(tmp_path / "render"))

        assert any(tmp_path.iterdir()), f"No animation was written for {env_name}"

    # TODO do I need to test rendering for generative stuff as well or is it assumed to work if the gen test passes?
