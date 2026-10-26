"""
Physical validity of the continuous time chaotic systems against their textbook definitions and published
Lyapunov exponents.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from bifurcagym.envs.continuous_time_chaos import chua, lorenz63, rossler
from physics_utils import lyapunov_spectrum, random_states, rk4_integrate


pytestmark = pytest.mark.physics


# # # Textbook vector fields, written independently of the envs
########################################################################################################################
def lorenz_rhs(params):
    def f(t, s):
        x, y, z = s
        return jnp.array((params.sigma * (y - x), x * (params.rho - z) - y, x * y - params.beta * z))
    return f


def rossler_rhs(params):
    def f(t, s):
        x, y, z = s
        return jnp.array((-y - z, x + params.a * y, params.b + z * (x - params.c)))
    return f


def chua_rhs(params):
    def f(t, s):
        x, y, z = s
        # Piecewise linear diode with slope m0 inside |x| < 1 and m1 outside
        diode = jnp.where(jnp.abs(x) < 1.0, params.m0 * x, params.m1 * x + jnp.sign(x) * (params.m0 - params.m1))
        return jnp.array((params.alpha * (y - x - diode), x - y + z, -params.beta * y))
    return f


CASES = {"Lorenz63": (lorenz63.Lorenz63CSCA, lorenz_rhs, dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0)),
         "Rossler": (rossler.RosslerCSCA, rossler_rhs, dict(a=0.2, b=0.2, c=5.7)),
         "Chua": (chua.ChuaCSCA, chua_rhs, dict(alpha=15.6, beta=28.0, m0=-1.143, m1=-0.714)),
         }

# Points spread over each attractor's extent
BOXES = {"Lorenz63": ((-20.0, -25.0, 0.0), (20.0, 25.0, 50.0)),
         "Rossler": ((-10.0, -10.0, 0.0), (12.0, 10.0, 20.0)),
         "Chua": ((-2.5, -0.5, -4.0), (2.5, 0.5, 4.0)),
         }


def env_rhs(env, params):
    return lambda t, s: env._f(t, s, jnp.zeros(3), params)


def zero_action(env, params):
    return jnp.zeros(env.action_space(params).shape)


@pytest.mark.parametrize("case", CASES.keys())
def test_default_parameters_are_the_canonical_chaotic_values(case):
    env_cls, _, canonical = CASES[case]
    params = env_cls().default_params
    for name, value in canonical.items():
        np.testing.assert_allclose(getattr(params, name), value, rtol=1e-12, err_msg=name)


@pytest.mark.parametrize("case", CASES.keys())
def test_vector_field_matches_textbook_definition(case):
    env_cls, rhs, _ = CASES[case]
    env = env_cls()
    params = env.default_params
    points = random_states(jrandom.key(0), 256, *BOXES[case])

    np.testing.assert_allclose(jax.vmap(lambda s: env_rhs(env, params)(0.0, s))(points),
                               jax.vmap(lambda s: rhs(params)(0.0, s))(points), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", CASES.keys())
def test_uncontrolled_rollout_matches_reference_integration(case):
    """Over roughly one Lyapunov time the env's integrator must track a fine independent RK4 solution"""
    env_cls, rhs, _ = CASES[case]
    env = env_cls()
    params = env.default_params
    obs, state = env.reset(params, jrandom.key(1))
    n_steps = int(round(1.0 / env.dt))
    refine = 50

    def body(state, key):
        obs, _, new_state, _, _, _ = env.step(zero_action(env, params), state, params, key)
        return new_state, obs

    _, env_traj = jax.lax.scan(body, state, jrandom.split(jrandom.key(2), n_steps))
    reference = rk4_integrate(rhs(params), obs, 0.0, env.dt / refine, n_steps * refine)[refine::refine]

    np.testing.assert_allclose(env_traj, reference, rtol=1e-6, atol=1e-6)


def test_lorenz_volume_contraction_and_symmetry():
    """div f = -(sigma + 1 + beta) everywhere, and f commutes with (x, y, z) -> (-x, -y, z)"""
    env = lorenz63.Lorenz63CSCA()
    params = env.default_params
    f = lambda s: env_rhs(env, params)(0.0, s)
    points = random_states(jrandom.key(3), 256, *BOXES["Lorenz63"])
    flip = jnp.array((-1.0, -1.0, 1.0))

    divergence = jax.vmap(lambda s: jnp.trace(jax.jacfwd(f)(s)))(points)
    np.testing.assert_allclose(divergence, -(params.sigma + 1 + params.beta), rtol=1e-12)
    np.testing.assert_allclose(jax.vmap(f)(points * flip), jax.vmap(f)(points) * flip, rtol=1e-12, atol=1e-12)


def test_chua_vector_field_is_odd():
    env = chua.ChuaCSCA()
    params = env.default_params
    f = lambda s: env_rhs(env, params)(0.0, s)
    points = random_states(jrandom.key(4), 256, *BOXES["Chua"])
    np.testing.assert_allclose(jax.vmap(f)(-points), -jax.vmap(f)(points), rtol=1e-12, atol=1e-12)


def test_rossler_fixed_point_is_an_equilibrium():
    env = rossler.RosslerCSCA()
    params = env.default_params
    np.testing.assert_allclose(env_rhs(env, params)(0.0, params.fixed_point), 0.0, atol=1e-10)


@pytest.mark.parametrize("case", ["Lorenz63", "Rossler"])
def test_integrator_converges_at_fourth_order(case):
    """
    Global error over a fixed time drops ~16x per halving of the RK4 substep. Chua is excluded as its diode is only
    piecewise smooth, which caps the order, its accuracy is covered by the reference rollout test instead
    """
    env_cls, rhs, _ = CASES[case]
    errors = []
    for factor in [1, 2]:
        env = env_cls()  # new env each time as step is jitted with the env as a static argument
        env.substeps *= factor
        params = env.default_params
        obs, state = env.reset(params, jrandom.key(5))
        n_steps = int(round(1.0 / env.dt))

        def body(state, key):
            obs, _, new_state, _, _, _ = env.step(zero_action(env, params), state, params, key)
            return new_state, obs

        _, env_traj = jax.lax.scan(body, state, jrandom.split(jrandom.key(6), n_steps))
        refine = 200
        reference = rk4_integrate(rhs(params), obs, 0.0, env.dt / refine, n_steps * refine)[refine::refine]
        errors.append(float(jnp.max(jnp.abs(env_traj - reference))))

    order = np.log2(errors[0] / errors[1])
    assert order > 3.5, f"Observed convergence order {order:.2f} from errors {errors}"


# # # Lyapunov exponents per unit time against published values
########################################################################################################################
LYAPUNOV_CASES = {
    # case: (number of env steps, expected largest exponent, tolerance, expected sum or None)
    "Lorenz63": (100_000, 0.9056, 0.05, None),  # Sprott (2003); sum is -(sigma + 1 + beta)
    "Rossler": (250_000, 0.0714, 0.015, None),  # Sprott (2003)
    "Chua": (100_000, None, None, None),  # only require chaos, published values vary with the scaling used
}


@pytest.mark.parametrize("case", LYAPUNOV_CASES.keys())
def test_lyapunov_exponents_match_literature(case):
    env_cls, _, _ = CASES[case]
    env = env_cls()
    params = env.default_params
    n_steps, expected_max, tol, _ = LYAPUNOV_CASES[case]
    obs, _ = env.reset(params, jrandom.key(7))
    action = zero_action(env, params)
    key = jrandom.key(8)

    def step_fn(x):
        return env.generative_step(action, x, params, key)[0]

    exponents = np.asarray(lyapunov_spectrum(step_fn, obs, n_steps, n_transient=int(50 / env.dt))) / env.dt

    if expected_max is None:
        assert exponents[0] > 0.1, f"{case} is not chaotic, exponents {exponents}"
    else:
        np.testing.assert_allclose(exponents[0], expected_max, atol=tol, err_msg=f"exponents {exponents}")
    np.testing.assert_allclose(exponents[1], 0.0, atol=0.05, err_msg="flows have one zero exponent")
    if case == "Lorenz63":
        np.testing.assert_allclose(exponents.sum(), -(params.sigma + 1 + params.beta), rtol=1e-2)
