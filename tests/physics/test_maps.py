"""
Physical validity of the discrete time chaotic maps against their textbook definitions and known Lyapunov exponents.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from bifurcagym import analysis
from bifurcagym.envs.discrete_time_chaos import (cat_map, henon_map, ikeda_map, logistic_map, lozi_map, skew_product,
                                                 standard_map, tent_map)
from physics_utils import lyapunov_spectrum, random_states


GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


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


def lozi(params):
    return lambda s: jnp.array((1 - params.init_a * jnp.abs(s[0]) + s[1], params.init_b * s[0]))


def logistic(r):
    return lambda s: r * s * (1 - s)


def tent(mu):
    return lambda s: mu * jnp.minimum(s, 1 - s)


def cat():
    return lambda s: jnp.mod(jnp.array(((2, 1), (1, 1))) @ s, 1.0)


def standard(K):
    def f(s):
        p = s[1] + K * jnp.sin(s[0])
        return jnp.mod(jnp.array((s[0] + p, p)) + jnp.pi, 2 * jnp.pi) - jnp.pi
    return f


def skew(params):
    return lambda s: jnp.array((params.a * s[0] + s[1], params.mu * jnp.minimum(s[1], 1 - s[1])))


def torus_delta(period):
    """a - b on a torus, for finite difference tangents that straddle the wrap"""
    return lambda a, b: (a - b) - period * jnp.round((a - b) / period)


def make_cases():
    henon_env = henon_map.HenonMapCSCA()
    ikeda_env = ikeda_map.IkedaMapCSCA()
    logistic_env = logistic_map.LogisticMapCSCA()
    tent_env = tent_map.TentMapCSCA()
    lozi_env = lozi_map.LoziMapCSCA()
    cat_env = cat_map.CatMapCSCA()
    standard_env = standard_map.StandardMapCSCA()
    skew_env = skew_product.SkewProductCSCA()
    return {
        # case: (env, textbook map, sample box)
        "HenonMap": (henon_env, henon(henon_env.default_params), ((-1.4, -0.4), (1.4, 0.4))),
        "LoziMap": (lozi_env, lozi(lozi_env.default_params), ((-1.2, -0.6), (1.2, 0.6))),
        "IkedaMap": (ikeda_env, ikeda(ikeda_env.default_params), ((-0.5, -2.5), (2.0, 1.0))),
        "LogisticMap": (logistic_env, logistic(logistic_env.init_r), ((0.0,), (1.0,))),
        "TentMap": (tent_env, tent(tent_env.default_params.init_mu), ((0.0,), (1.0,))),
        "CatMap": (cat_env, cat(), ((0.0, 0.0), (1.0, 1.0))),
        "StandardMap": (standard_env, standard(standard_env.default_params.K), ((-np.pi, -np.pi), (np.pi, np.pi))),
        "SkewProduct": (skew_env, skew(skew_env.default_params), ((-1.0, 0.0), (2.0, 1.0))),
    }


FIXED_POINT_CASES = [case for case in make_cases() if case != "SkewProduct"]  # its target is x = 0 whatever w is


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
    np.testing.assert_allclose((lozi_map.EnvParams().init_a, lozi_map.EnvParams().init_b), (1.7, 0.5))
    ikeda_params = ikeda_map.EnvParams()
    np.testing.assert_allclose((ikeda_params.u_param, ikeda_params.k, ikeda_params.p), (0.9, 0.4, 6.0))


@pytest.mark.parametrize("case", FIXED_POINT_CASES)
def test_fixed_point_is_a_fixed_point(case):
    env, expected_map, _ = make_cases()[case]
    params = env.default_params
    fixed_point = params.fixed_point if hasattr(params, "fixed_point") else env.fixed_point
    fixed_point = jnp.atleast_1d(jnp.asarray(fixed_point, dtype=jnp.float64))

    np.testing.assert_allclose(expected_map(fixed_point), fixed_point, atol=1e-8)


def test_henon_ikeda_and_lozi_jacobian_determinants():
    """det J is -b for Henon and Lozi (either side of Lozi's kink at x = 0) and u^2 for Ikeda everywhere"""
    points = random_states(jrandom.key(1), 256, (-1.0, -1.0), (1.0, 1.0))
    henon_params = henon_map.EnvParams()
    ikeda_params = ikeda_map.EnvParams()
    lozi_params = lozi_map.EnvParams()

    henon_det = jax.vmap(lambda s: jnp.linalg.det(jax.jacfwd(henon(henon_params))(s)))(points)
    ikeda_det = jax.vmap(lambda s: jnp.linalg.det(jax.jacfwd(ikeda(ikeda_params))(s)))(points)
    lozi_det = jax.vmap(lambda s: jnp.linalg.det(jax.jacfwd(lozi(lozi_params))(s)))(points)
    np.testing.assert_allclose(henon_det, -henon_params.init_b, rtol=1e-10)
    np.testing.assert_allclose(ikeda_det, ikeda_params.u_param ** 2, rtol=1e-10)
    np.testing.assert_allclose(lozi_det, -lozi_params.init_b, rtol=1e-10)


def test_lozi_attractor_is_hyperbolic():
    """Misiurewicz (1980): the expanding and contracting directions never become tangent"""
    env = lozi_map.LoziMapCSCA()
    params = env.default_params
    _, state = env.reset(params, jrandom.key(1))
    check = analysis.hyperbolicity(env, params, state, n_steps=50_000)

    assert check.verdict == "hyperbolic" and check.min_angle > 0.3


# # # Conservative maps on the torus
########################################################################################################################
@pytest.mark.parametrize("case", ["CatMap", "StandardMap"])
def test_torus_maps_preserve_area(case):
    """det J = 1 everywhere, so there is no attractor"""
    env, expected_map, box = make_cases()[case]
    points = random_states(jrandom.key(1), 256, *box)

    dets = jax.vmap(lambda s: jnp.linalg.det(jax.jacfwd(uncontrolled_map(env, env.default_params))(s)))(points)
    np.testing.assert_allclose(dets, 1.0, rtol=1e-10)


def test_cat_map_is_hyperbolic_with_perpendicular_directions():
    """The Jacobian is symmetric, so the stable and unstable directions are its orthogonal eigenvectors everywhere"""
    env = cat_map.CatMapCSCA()
    params = env.default_params
    _, state = env.reset(params, jrandom.key(1))
    check = analysis.hyperbolicity(env, params, state, n_steps=20_000)

    assert check.verdict == "hyperbolic"
    np.testing.assert_allclose(check.min_angle, np.pi / 2, atol=1e-3)


def test_standard_map_has_islands_the_sea_never_enters():
    """
    At K = 1 the main island around the elliptic point (pi, 0) is regular (lambda ~ 0) and orbits started from the
    reset box, in the chaotic sea, never enter it. Whether an orbit is chaotic depends on where it starts.
    """
    env = standard_map.StandardMapCSCA()
    params = env.default_params
    f = standard(params.K)
    centre = jnp.array((jnp.pi, 0.0))

    jacobian_trace = jnp.trace(jax.jacfwd(f)(centre - jnp.array((1e-12, 0.0))))
    assert abs(float(jacobian_trace)) < 2, "(pi, 0) should be elliptic"

    def orbit(s0, n):
        def body(s, _):
            s = f(s)
            return s, s

        return jax.lax.scan(body, s0, None, n)[1]

    distance = jax.vmap(lambda s: cat_map.torus_distance(s, centre, 2 * jnp.pi))
    island = orbit(jnp.array((jnp.pi - 0.3, 0.0)), 20_000)
    sea = jax.vmap(lambda key: orbit(env.reset(params, key)[0], 20_000))(jrandom.split(jrandom.key(5), 16))

    assert float(jnp.max(distance(island))) < 0.5
    assert float(jnp.min(jax.vmap(distance)(sea))) > 0.3

    island_lyapunov = lyapunov_spectrum(f, island[0], n_steps=100_000, n_transient=0, d0=1e-9,
                                        delta_fn=torus_delta(2 * np.pi))
    sea_lyapunov = lyapunov_spectrum(f, sea[0, 0], n_steps=100_000, n_transient=0, d0=1e-9,
                                     delta_fn=torus_delta(2 * np.pi))
    assert abs(float(island_lyapunov[0])) < 0.01 and float(sea_lyapunov[0]) > 0.05


# # # Skew product, a chaotic driver the agent cannot change
########################################################################################################################
def test_skew_product_observed_driver_is_cancelled_in_one_step():
    """u = -(a x + w) puts x on 0 in one step and keeps it there, the sanity check for the observed variant"""
    env = skew_product.SkewProductCSCA()
    params = env.default_params

    def episode(key):
        _, state = env.reset(params, key)

        def body(state, key):
            action = jnp.atleast_1d(-(params.a * state.x + state.w))
            _, _, state, reward, _, _ = env.step(action, state, params, key)
            return state, reward

        return jax.lax.scan(body, state, jrandom.split(key, 200))[1]

    rewards = jax.vmap(episode)(jrandom.split(jrandom.key(6), 64))
    np.testing.assert_allclose(rewards, 1.0, atol=1e-12)


def test_skew_product_hidden_driver_only_shows_x():
    env = skew_product.SkewProductCSCA(observe_driver=False)
    params = env.default_params
    obs, state = env.reset(params, jrandom.key(0))

    assert obs.shape == (1,) and env.observation_space(params).shape == (1,)
    with pytest.raises(ValueError, match="No Generative Step"):
        env.generative_step(jnp.zeros(1), obs, params, jrandom.key(0))

    # The driver can be inferred from x's history: w_t = x_{t+1} - a x_t - u_t
    action = jnp.array((0.3,))
    next_obs, _, _, _, _, _ = env.step(action, state, params, jrandom.key(1))
    np.testing.assert_allclose(next_obs[0] - params.a * obs[0] - action[0], state.w, atol=1e-12)


# # # Lyapunov exponents per iteration
########################################################################################################################
LYAPUNOV_CASES = {
    # case: (env factory, params changes, expected spectrum, tolerance)
    "HenonMap": (henon_map.HenonMapCSCA, {}, (0.41922, -1.62319), 0.01),  # Sprott (2003)
    "IkedaMap": (ikeda_map.IkedaMapCSCA, {}, (0.5076, 2 * np.log(0.9) - 0.5076), 0.01),  # sum is 2 ln u exactly
    "LogisticMap": (lambda: logistic_map.LogisticMapCSCA(init_r=4.0), {}, (np.log(2.0),), 0.01),
    "TentMap": (tent_map.TentMapCSCA, dict(init_mu=1.9), (np.log(1.9),), 0.01),  # |f'| = mu everywhere
    # Regression value from two resets (0.4697, 0.4705), the sum is ln b exactly as |det J| = b everywhere
    "LoziMap": (lozi_map.LoziMapCSCA, {}, (0.470, np.log(0.5) - 0.470), 0.01),
    # Closed form, the eigenvalues of ((2, 1), (1, 1)) are phi^2 and 1 / phi^2
    "CatMap": (cat_map.CatMapCSCA, {}, (2 * np.log(GOLDEN_RATIO), -2 * np.log(GOLDEN_RATIO)), 0.001),
    # Closed form, the tent driver stretches by mu everywhere and x contracts by a
    "SkewProduct": (skew_product.SkewProductCSCA, {}, (np.log(1.99), np.log(0.5)), 0.001),
}
TORUS_PERIODS = {"CatMap": 1.0}  # finite difference tangents need the torus distance across the wrap


@pytest.mark.parametrize("case", LYAPUNOV_CASES.keys())
def test_lyapunov_exponents_match_known_values(case):
    env_factory, param_changes, expected, tol = LYAPUNOV_CASES[case]
    env = env_factory()
    params = env.default_params.replace(**param_changes)
    obs, _ = env.reset(params, jrandom.key(2))
    delta_fn = torus_delta(TORUS_PERIODS[case]) if case in TORUS_PERIODS else None

    exponents = lyapunov_spectrum(uncontrolled_map(env, params), obs, n_steps=200_000, n_transient=1000, d0=1e-9,
                                  delta_fn=delta_fn)
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


@pytest.mark.parametrize("case", ["HenonMap", "LoziMap"])
@pytest.mark.parametrize("bang_bang", [False, True])
def test_random_control_at_full_range_does_not_escape(case, bang_bang):
    """
    Both maps sit near a crisis, so a control range that is too wide lets random exploration push every orbit off the
    attractor (it did for Henon at +-0.1 on a and b, and for Lozi at +-0.1 on a)
    """
    env, _, _ = make_cases()[case]
    params = env.default_params
    space = env.action_space(params)
    high = jnp.broadcast_to(jnp.asarray(space.high), space.shape)  # scalar or per action bounds

    def episode(key):
        reset_key, key = jrandom.split(key)
        _, state = env.reset(params, reset_key)

        def body(carry, key):
            state, alive = carry
            if bang_bang:
                action = jrandom.choice(key, jnp.array((-1.0, 0.0, 1.0)), high.shape) * high
            else:
                action = jrandom.uniform(key, high.shape, minval=-high, maxval=high)
            _, _, state, _, done, _ = env.step(action, state, params, key)
            return (state, alive & ~(done & (state.time < env.max_steps_in_ep))), None

        (_, alive), _ = jax.lax.scan(body, (state, jnp.array(True)), jrandom.split(key, env.max_steps_in_ep))
        return alive

    survived = float(jnp.mean(jax.vmap(episode)(jrandom.split(jrandom.key(4), 512))))
    assert survived > 0.98, f"{case}: only {survived:.1%} of random control episodes avoid escaping"
