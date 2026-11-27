"""
Physical validity of the Kuramoto-Sivashinsky PDE and Wet Chicken. The AYS integrated assessment model's tests moved
to experimental/other_envs/test_ays_iam.py with the env.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from bifurcagym.envs.classical_control import wet_chicken
from bifurcagym.envs import utils
from bifurcagym.envs.continuous_time_chaos import kuramoto_sivashinsky
from physics_utils import lyapunov_spectrum


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


def test_ks_solvers_converge_to_the_same_solution():
    """
    From a state on the attractor, over 1 time unit: every solver approaches a fine ETDRK4 reference, the default IMEX
    RK3 at second order and ETDRK4 at (at least) fourth order as substeps are refined
    """
    env = kuramoto_sivashinsky.KuramotoSivashinskyCSCA()
    params = env.default_params
    _, state = env.reset(params, jrandom.key(0))
    on_attractor, _ = ks_rollout(env, params, state, 4000)  # 2000 steps can still sit on the reset equilibrium

    def after(solver, substeps):
        solver_env = kuramoto_sivashinsky.KuramotoSivashinskyCSCA(solver=solver, substeps=substeps)
        return ks_rollout(solver_env, params, on_attractor, 20)[0].u

    imex, etdrk4, rk4 = utils.imex_rk3_step, utils.etdrk4_step, utils.semilinear_rk4_step
    reference = after(etdrk4, 64)
    error = {(solver, substeps): float(jnp.max(jnp.abs(after(solver, substeps) - reference)))
             for solver, substeps in ((imex, 1), (imex, 4), (etdrk4, 1), (etdrk4, 4), (rk4, 256))}

    # Errors this small on an equilibrium would make the order checks vacuous, so require a moving state
    assert error[(imex, 1)] > 1e-9, "still on the reset equilibrium, the convergence check is meaningless"
    assert error[(imex, 1)] < 1e-5 and error[(etdrk4, 1)] < 1e-6 and error[(rk4, 256)] < 1e-10
    assert error[(imex, 4)] < error[(imex, 1)] / 4 ** 1.8  # second order
    assert error[(etdrk4, 4)] < error[(etdrk4, 1)] / 4 ** 3  # fourth order, with some slack


def test_ks_default_solver_is_unchanged():
    env = kuramoto_sivashinsky.KuramotoSivashinskyCSCA()
    assert (env.solver, env.substeps) == (utils.imex_rk3_step, 1)


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
