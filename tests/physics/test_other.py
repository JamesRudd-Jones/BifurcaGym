"""
Physical validity of the Kuramoto-Sivashinsky PDE, the AYS integrated assessment model and Wet Chicken.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from bifurcagym.envs.classical_control import wet_chicken
from bifurcagym.envs.continuous_time_chaos import kuramoto_sivashinsky
from bifurcagym.envs.other_envs import ays_iam
from physics_utils import lyapunov_spectrum, random_states


pytestmark = pytest.mark.physics


# # # Kuramoto-Sivashinsky: u_t + u u_x + u_xx + u_xxxx = f on a periodic domain
########################################################################################################################
def ks_rhs(u, length):
    """Unforced KS tendency evaluated spectrally, written independently of the env's solver"""
    n = u.shape[-1]
    k = 2 * jnp.pi / length * jnp.fft.rfftfreq(n, d=1.0 / n)
    u_k = jnp.fft.rfft(u)
    return jnp.fft.irfft(-0.5j * k * jnp.fft.rfft(u ** 2) + (k ** 2 - k ** 4) * u_k, n=n)


def ks_rollout(env, params, state, n_steps):
    action = jnp.zeros(env.action_space(params).shape)

    def body(state, key):
        _, _, new_state, _, _, _ = env.step(action, state, params, key)
        return new_state, new_state.u

    return jax.lax.scan(body, state, jrandom.split(jrandom.key(0), n_steps))


@pytest.mark.parametrize("solution", ["u1", "u2", "u3"])
def test_ks_target_solutions_are_equilibria(solution):
    """The bundled target solutions are the L = 22 equilibria (Bucci et al. 2019), so the unforced tendency vanishes"""
    env = kuramoto_sivashinsky.KuramotoSivashinskyCSCA()
    u = jnp.asarray(np.loadtxt(env.data_dir / f"{solution}.dat"))
    np.testing.assert_array_less(jnp.linalg.norm(ks_rhs(u, env.L)) / jnp.linalg.norm(u), 1e-4)


def test_ks_solver_matches_unforced_tendency():
    """For a short step from a smooth state the solver's update approximates dt * (u_t)"""
    env = kuramoto_sivashinsky.KuramotoSivashinskyCSCA()
    params = env.default_params
    _, state = env.reset(params, jrandom.key(0))
    final_state, _ = ks_rollout(env, params, state, 2000)  # leave the equilibrium onto the chaotic attractor

    _, traj = ks_rollout(env, params, final_state, 1)
    expected = final_state.u + env.dt * ks_rhs(final_state.u, env.L)
    # first order in dt from the forward difference, the tendency itself is O(1)
    np.testing.assert_allclose(traj[0], expected, atol=5 * env.dt ** 2 * float(jnp.max(jnp.abs(ks_rhs(final_state.u, env.L)))) + 1e-3)


def test_ks_conserves_mean_and_stays_bounded():
    env = kuramoto_sivashinsky.KuramotoSivashinskyCSCA()
    params = env.default_params
    _, state = env.reset(params, jrandom.key(0))
    _, traj = ks_rollout(env, params, state, 20_000)

    np.testing.assert_allclose(jnp.mean(traj, axis=-1), jnp.mean(state.u), atol=1e-10)
    assert jnp.all(jnp.isfinite(traj)) and float(jnp.max(jnp.abs(traj))) < 5.0


def test_ks_is_chaotic():
    """Leading Lyapunov exponent for L = 22 is about 0.043 (Cvitanovic, Davidchack and Siminos 2010)"""
    env = kuramoto_sivashinsky.KuramotoSivashinskyCSCA()
    params = env.default_params
    _, state = env.reset(params, jrandom.key(0))
    action = jnp.zeros(env.action_space(params).shape)

    def step_fn(u):
        return env.step(action, state.replace(u=u), params, jrandom.key(0))[2].u

    exponents = lyapunov_spectrum(step_fn, state.u, n_steps=40_000, n_transient=4_000, n_exponents=2) / env.dt
    np.testing.assert_allclose(exponents[0], 0.043, atol=0.015)


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


@pytest.mark.parametrize("param_changes", [{}, dict(sigma=2e12, rho=3.0, beta=0.025)])
def test_ays_matches_published_model(param_changes):
    env = ays_iam.AYSIAMCSCA()
    params = env.default_params.replace(**param_changes)
    points = random_states(jrandom.key(0), 256, 0.05 * jnp.ones(3), 0.95 * jnp.ones(3))

    for action in env.action_array:
        actual = jax.vmap(lambda s: env._ays_rhs(0.0, s, action, params))(points)
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


# # # Wet Chicken: Hans and Udluft (2009), drift v = 3x / w and turbulence s = 3.5 - v
########################################################################################################################
def test_wet_chicken_drift_and_turbulence_statistics():
    env = wet_chicken.WetChickenCSCA()
    params = env.default_params
    x, y = 4.0, 0.5  # chosen so no sample hits the waterfall or the bank
    velocity = params.max_velocity * x / env.width
    turbulence = params.max_turbulence - velocity
    state = wet_chicken.EnvState(x=jnp.array(x), y=jnp.array(y), time=0)
    action = jnp.array((0.0, -0.5))

    new_y = jax.vmap(lambda key: env.step(action, state, params, key)[2].y)(jrandom.split(jrandom.key(0), 20_000))
    dy = new_y - y

    np.testing.assert_allclose(jnp.mean(dy), action[1] + velocity, atol=0.03)
    np.testing.assert_allclose(jnp.var(dy), turbulence ** 2 / 3, rtol=0.05)  # uniform on [-s, s]
    np.testing.assert_allclose((jnp.min(dy), jnp.max(dy)), (action[1] + velocity - turbulence,
                                                            action[1] + velocity + turbulence), atol=0.01)
