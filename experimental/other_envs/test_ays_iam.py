"""
Physical validity of the AYS integrated assessment model, moved from tests/physics/test_other.py when the env moved
to experimental/. Not collected by the suite (experimental/ is outside testpaths), run by hand with

    JAX_PLATFORMS=cpu pdm run pytest experimental/other_envs/test_ays_iam.py -m physics
"""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import ays_iam  # noqa: E402


jax.config.update("jax_enable_x64", True)
pytestmark = pytest.mark.physics


def random_states(key, n: int, low, high) -> jnp.ndarray:
    low = jnp.asarray(low, dtype=jnp.float64)
    high = jnp.asarray(high, dtype=jnp.float64)
    return jrandom.uniform(key, (n,) + low.shape, minval=low, maxval=high, dtype=jnp.float64)


# # # AYS: Kittel et al. (2017) model in physical units, then compactified to [0, 1]^3
########################################################################################################################
A_MID, Y_MID, S_MID = 240.0, 7e13, 5e11


def ays_physical_rhs(params, action):
    """dA/dt = Gamma Y / (eps phi) - A / tau_A, dY/dt = beta Y - theta A Y, dS/dt = (1 - Gamma) Y / eps - S / tau_S"""
    beta = params.beta * action[0]  # low growth halves beta
    sigma = params.sigma * 0.5 ** (action[1] / params.rho)  # energy transition lowers the renewable break even cost

    def f(phys):
        a, y, s = phys
        gamma = 1.0 / (1.0 + (s / sigma) ** params.rho)
        return jnp.array((gamma * y / (params.eps * params.phi) - a / params.tau_A,
                          beta * y - params.theta * a * y,
                          (1.0 - gamma) * y / params.eps - s / params.tau_S))
    return f


def to_physical(ays):
    mids = jnp.array((A_MID, Y_MID, S_MID))
    return mids * ays / (1.0 - ays)


def to_compact(phys):
    mids = jnp.array((A_MID, Y_MID, S_MID))
    return phys / (phys + mids)


def ays_compact_rhs(params, action):
    f = ays_physical_rhs(params, action)

    def g(ays):
        phys = to_physical(ays)
        _, compact_rate = jax.jvp(to_compact, (phys,), (f(phys),))
        return compact_rate
    return g


@pytest.mark.parametrize("param_changes", [{}, {"sigma": 2e12, "rho": 3.0, "beta": 0.025}])
def test_ays_matches_published_model(param_changes):
    env = ays_iam.AYSIAMCSCA()
    params = env.default_params.replace(**param_changes)
    points = random_states(jrandom.key(0), 256, 0.05 * jnp.ones(3), 0.95 * jnp.ones(3))

    for action in env.action_array:
        actual = jax.vmap(lambda s, action=action: env._ays_rhs(0.0, s, action, params))(points)
        expected = jax.vmap(ays_compact_rhs(params, action))(points)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, err_msg=f"action {action}, params {param_changes}")


def test_ays_stays_in_the_unit_cube():
    env = ays_iam.AYSIAMCSDA()
    params = env.default_params

    def episode(key):
        reset_key, action_key = jrandom.split(key)
        _, state = env.reset(params, reset_key)

        def body(state, key):
            action_key, step_key = jrandom.split(key)
            action = env.action_space(params).sample(action_key)
            _, _, new_state, _, _, _ = env.step(action, state, params, step_key)
            return new_state, new_state.ays

        return jax.lax.scan(body, state, jrandom.split(action_key, 500))[1]

    trajs = jax.vmap(episode)(jrandom.split(jrandom.key(1), 64))
    assert jnp.all(jnp.isfinite(trajs)) and jnp.all(trajs >= 0.0) and jnp.all(trajs <= 1.0)


def test_ays_default_policy_reaches_the_black_fixed_point():
    """Business as usual (no management) should end at the high carbon, low renewables attractor"""
    env = ays_iam.AYSIAMCSDA()
    params = env.default_params
    _, state = env.reset(params, jrandom.key(2))
    action = jnp.array((0,))  # no management option

    def body(state, key):
        _, _, new_state, _, _, _ = env.step(action, state, params, key)
        return new_state, None

    final_state, _ = jax.lax.scan(body, state, jrandom.split(jrandom.key(3), 3000))
    assert float(jnp.linalg.norm(final_state.ays - params.black_fp)) < params.final_radius
