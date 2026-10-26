"""
Physical validity of the discrete time chaotic maps against their textbook definitions and known Lyapunov exponents.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from bifurcagym.envs.discrete_time_chaos import henon_map, ikeda_map, logistic_map, tent_map, tinkerbell_map
from physics_utils import lyapunov_spectrum, random_states


pytestmark = pytest.mark.physics


# # # Textbook maps, written independently of the envs
########################################################################################################################
def henon(params):
    return lambda s: jnp.array((1 - params.init_a * s[0] ** 2 + s[1], params.init_b * s[0]))


def ikeda(params):
    def f(s):
        t = params.k - params.p / (1 + s[0] ** 2 + s[1] ** 2)
        return jnp.array((1 + params.u_param * (s[0] * jnp.cos(t) - s[1] * jnp.sin(t)),
                          params.u_param * (s[0] * jnp.sin(t) + s[1] * jnp.cos(t))))
    return f


def tinkerbell(params):
    return lambda s: jnp.array((s[0] ** 2 - s[1] ** 2 + params.a * s[0] + params.b * s[1],
                                2 * s[0] * s[1] + params.c * s[0] + params.d * s[1]))


def logistic(r):
    return lambda s: r * s * (1 - s)


def tent(mu):
    return lambda s: mu * jnp.minimum(s, 1 - s)


def make_cases():
    henon_env = henon_map.HenonMapCSCA()
    ikeda_env = ikeda_map.IkedaMapCSCA()
    tinkerbell_env = tinkerbell_map.TinkerbellMapCSCA()
    logistic_env = logistic_map.LogisticMapCSCA()
    tent_env = tent_map.TentMapCSCA()
    return {
        # case: (env, textbook map, sample box)
        "HenonMap": (henon_env, henon(henon_env.default_params), ((-1.4, -0.4), (1.4, 0.4))),
        "IkedaMap": (ikeda_env, ikeda(ikeda_env.default_params), ((-0.5, -2.5), (2.0, 1.0))),
        "TinkerbellMap": (tinkerbell_env, tinkerbell(tinkerbell_env.default_params), ((-1.2, -1.5), (0.4, 0.5))),
        "LogisticMap": (logistic_env, logistic(logistic_env.init_r), ((0.0,), (1.0,))),
        "TentMap": (tent_env, tent(tent_env.default_params.init_mu), ((0.0,), (1.0,))),
    }


def uncontrolled_map(env, params):
    action = jnp.zeros(env.action_space(params).shape)
    return lambda s: env.generative_step(action, s, params, jrandom.key(0))[0]


@pytest.mark.parametrize("case", make_cases().keys())
def test_map_matches_textbook_definition(case):
    env, expected_map, box = make_cases()[case]
    params = env.default_params
    points = random_states(jrandom.key(0), 256, *box)

    np.testing.assert_allclose(jax.vmap(uncontrolled_map(env, params))(points), jax.vmap(expected_map)(points),
                               rtol=1e-12, atol=1e-12)


def test_default_parameters_are_the_canonical_chaotic_values():
    np.testing.assert_allclose((henon_map.EnvParams().init_a, henon_map.EnvParams().init_b), (1.4, 0.3))
    ikeda_params = ikeda_map.EnvParams()
    np.testing.assert_allclose((ikeda_params.u_param, ikeda_params.k, ikeda_params.p), (0.9, 0.4, 6.0))
    tinkerbell_params = tinkerbell_map.EnvParams()
    np.testing.assert_allclose((tinkerbell_params.a, tinkerbell_params.b, tinkerbell_params.c, tinkerbell_params.d),
                               (0.9, -0.6013, 2.0, 0.5))


@pytest.mark.parametrize("case", make_cases().keys())
def test_fixed_point_is_a_fixed_point(case):
    env, expected_map, _ = make_cases()[case]
    params = env.default_params
    fixed_point = params.fixed_point if hasattr(params, "fixed_point") else env.fixed_point
    fixed_point = jnp.atleast_1d(jnp.asarray(fixed_point, dtype=jnp.float64))

    np.testing.assert_allclose(expected_map(fixed_point), fixed_point, atol=1e-8)


def test_tinkerbell_fixed_point_is_the_unstable_focus_inside_the_attractor():
    """Of the two fixed points, the control target should be the one embedded in the attractor (the origin)"""
    env = tinkerbell_map.TinkerbellMapCSCA()
    f = tinkerbell(env.default_params)

    def body(s, _):
        return f(s), f(s)

    _, orbit = jax.lax.scan(body, jnp.array((-0.72, -0.64)), None, 20_000)
    eigs = np.linalg.eigvals(np.asarray(jax.jacfwd(f)(env.fixed_point)))

    assert np.min(np.linalg.norm(np.asarray(orbit[1000:]) - np.asarray(env.fixed_point), axis=1)) < 0.2
    assert np.all(np.abs(eigs) > 1) and np.all(np.abs(eigs.imag) > 0), f"Expected an unstable focus, got {eigs}"


def test_henon_and_ikeda_jacobian_determinants():
    """det J is -b for Henon and u^2 for Ikeda everywhere"""
    points = random_states(jrandom.key(1), 256, (-1.0, -1.0), (1.0, 1.0))
    henon_params = henon_map.EnvParams()
    ikeda_params = ikeda_map.EnvParams()

    henon_det = jax.vmap(lambda s: jnp.linalg.det(jax.jacfwd(henon(henon_params))(s)))(points)
    ikeda_det = jax.vmap(lambda s: jnp.linalg.det(jax.jacfwd(ikeda(ikeda_params))(s)))(points)
    np.testing.assert_allclose(henon_det, -henon_params.init_b, rtol=1e-10)
    np.testing.assert_allclose(ikeda_det, ikeda_params.u_param ** 2, rtol=1e-10)


# # # Lyapunov exponents per iteration
########################################################################################################################
LYAPUNOV_CASES = {
    # case: (env factory, params changes, expected spectrum, tolerance)
    "HenonMap": (henon_map.HenonMapCSCA, {}, (0.41922, -1.62319), 0.01),  # Sprott (2003)
    "IkedaMap": (ikeda_map.IkedaMapCSCA, {}, (0.5076, 2 * np.log(0.9) - 0.5076), 0.01),  # sum is 2 ln u exactly
    "LogisticMap": (lambda: logistic_map.LogisticMapCSCA(init_r=4.0), {}, (np.log(2.0),), 0.01),
    "TentMap": (tent_map.TentMapCSCA, dict(init_mu=1.9), (np.log(1.9),), 0.01),  # |f'| = mu everywhere
}


@pytest.mark.parametrize("case", LYAPUNOV_CASES.keys())
def test_lyapunov_exponents_match_known_values(case):
    env_factory, param_changes, expected, tol = LYAPUNOV_CASES[case]
    env = env_factory()
    params = env.default_params.replace(**param_changes)
    obs, _ = env.reset(params, jrandom.key(2))

    exponents = lyapunov_spectrum(uncontrolled_map(env, params), obs, n_steps=200_000, n_transient=1000, d0=1e-9)
    np.testing.assert_allclose(exponents, expected, atol=tol)


# # # Reset distributions
########################################################################################################################
@pytest.mark.parametrize("case", make_cases().keys())
def test_uncontrolled_orbits_from_reset_stay_on_the_attractor(case):
    """Episodes should not end by divergence or numerical collapse before the agent has acted"""
    env, _, _ = make_cases()[case]
    params = env.default_params
    step = uncontrolled_map(env, params)

    def orbit(key):
        obs, _ = env.reset(params, key)

        def body(s, _):
            s = step(s)
            return s, s

        _, traj = jax.lax.scan(body, obs, None, 1000)
        return traj

    trajs = jax.vmap(orbit)(jrandom.split(jrandom.key(3), 512))
    bounded = jnp.all(jnp.isfinite(trajs) & (jnp.abs(trajs) < 10.0), axis=(1, 2))
    # A chaotic orbit keeps moving, a collapse onto a fixed point (e.g. the tent map in floating point) does not
    moving = jnp.std(trajs[:, -100:], axis=(1, 2)) > 1e-3

    assert float(jnp.mean(bounded)) > 0.99, f"{case}: only {float(jnp.mean(bounded)):.0%} of resets stay bounded"
    assert float(jnp.mean(moving)) > 0.99, f"{case}: only {float(jnp.mean(moving)):.0%} of orbits stay chaotic"
