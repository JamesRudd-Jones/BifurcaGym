"""Opt-in Double Gyre / ABC options: Lipschitz reward and random start (used in the L4DC 2027 experiments)"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

import bifurcagym

FLOWS = ["DoubleGyreFlow-v0", "ABCFlow-v0"]


def _make(env_name, **kwargs):
    env, env_params = bifurcagym.make(env_name, cont_state=True, cont_action=False, **kwargs)
    return env, env_params


def _dist_to_goal(env, state):
    if hasattr(env, "_torus_dist_to_goal"):
        return env._torus_dist_to_goal(jnp.array((state.x, state.y, state.z)))
    return jnp.linalg.norm(jnp.array((state.x, state.y)) - env.goal_state)


@pytest.mark.parametrize("env_name", FLOWS)
def test_random_start_in_bounds_and_away_from_goal(env_name):
    env, env_params = _make(env_name, random_start=True)
    keys = jrandom.split(jrandom.key(0), 256)
    _, states = jax.vmap(env.reset, in_axes=(None, 0))(env_params, keys)

    dists = jax.vmap(lambda s: _dist_to_goal(env, s))(states)
    assert jnp.all(dists >= env.start_exclusion)
    assert jnp.all((states.x >= env.x_bounds[0]) & (states.x <= env.x_bounds[1]))
    assert jnp.all((states.y >= env.y_bounds[0]) & (states.y <= env.y_bounds[1]))
    assert jnp.unique(states.x).size == keys.shape[0]  # starts actually differ across keys
    assert jnp.all(states.time == 0)


@pytest.mark.parametrize("env_name", FLOWS)
def test_default_start_unchanged(env_name):
    env, env_params = _make(env_name)
    _, state = env.reset(env_params, jrandom.key(0))
    assert jnp.allclose(state.x, 0.2, atol=1e-3) and jnp.allclose(state.y, 0.2, atol=1e-3)


@pytest.mark.parametrize("env_name", FLOWS)
def test_lipschitz_reward_is_minus_distance_and_never_terminates_early(env_name):
    env, env_params = _make(env_name, lipschitz_reward=True, random_start=True)
    key = jrandom.key(1)
    _, state = env.reset(env_params, key)

    for t in range(200):
        key, _key = jrandom.split(key)
        action = env.action_space(env_params).sample(_key)
        _, _, state, reward, done, _ = env.step(action, state, env_params, _key)
        assert jnp.allclose(reward, -_dist_to_goal(env, state))
        assert not done


@pytest.mark.parametrize("env_name", FLOWS)
def test_lipschitz_reward_done_at_time_limit(env_name):
    env, env_params = _make(env_name, lipschitz_reward=True)
    _, state = env.reset(env_params, jrandom.key(0))
    state = state.replace(time=env.max_steps_in_ep - 1)
    _, _, _, _, done, _ = env.step(jnp.array(0), state, env_params, jrandom.key(0))
    assert done


@pytest.mark.parametrize("env_name", FLOWS)
def test_unsupported_kwarg_raises(env_name):
    with pytest.raises(TypeError, match="unsupported env kwargs"):
        _make(env_name, random_strat=True)
