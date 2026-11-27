"""
Tests for the shared integrators in bifurcagym.envs.utils: integrate_explicit, integrate_semilinear and
integrate_implicit, each taking a solver (step function) from its own family.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bifurcagym.envs import utils


def linear_problem():
    """x' = L x + c with stiff diagonal L, exact solution x(t) = e^{Lt} x0 + (e^{Lt} - 1) c / L"""
    linear = jnp.array((-1.0, -50.0, -2_000.0, 0.5))
    forcing = jnp.array((1.0, -2.0, 3.0, 0.25))
    x0 = jnp.array((0.3, -0.1, 2.0, 1.0))

    def exact(t):
        growth = jnp.exp(linear * t)
        return growth * x0 + (growth - 1.0) * forcing / linear

    return linear, forcing, x0, exact


def constant(t, x, u, p):
    return u


def forced_decay(t, x, u, p):
    """x' = -x + sin(t), smooth and non-stiff, for convergence orders"""
    return -x + jnp.sin(t)


def forced_decay_exact(t, x0):
    return (x0 + 0.5) * jnp.exp(-t) + 0.5 * (jnp.sin(t) - jnp.cos(t))


def observed_order(integrate, solver, f=forced_decay, exact=forced_decay_exact, n_steps=(20, 40)):
    x0 = jnp.array((1.0,))
    errors = [float(jnp.abs(integrate(f, 0.0, x0, None, 2.0, n, None, solver=solver) - exact(2.0, x0))[0])
              for n in n_steps]
    return np.log2(errors[0] / errors[1])


# # # integrate_explicit
########################################################################################################################
@pytest.mark.parametrize("solver, order", [(utils.euler_step, 1), (utils.rk4_step, 4)])
def test_explicit_convergence_order(solver, order):
    """
    On pure decay x' = -x, where RK4's error is in its asymptotic regime from the start (with the sin forcing it only
    reaches fourth order at much smaller steps)
    """
    decay = lambda t, x, u, p: -x  # noqa: E731
    decay_exact = lambda t, x0: x0 * jnp.exp(-t)  # noqa: E731
    assert observed_order(utils.integrate_explicit, solver, decay, decay_exact) > order - 0.2


def test_explicit_default_is_rk4():
    x0 = jnp.array((1.0,))
    default = utils.integrate_explicit(forced_decay, 0.0, x0, None, 1.0, 10, None)
    rk4 = utils.integrate_explicit(forced_decay, 0.0, x0, None, 1.0, 10, None, solver=utils.rk4_step)
    np.testing.assert_array_equal(default, rk4)


# # # integrate_semilinear
########################################################################################################################
def test_etdrk4_is_exact_for_a_constant_nonlinear_term():
    linear, forcing, x0, exact = linear_problem()
    x = utils.integrate_semilinear(linear, constant, 0.0, x0, forcing, 0.5, 1, None, solver=utils.etdrk4_step)
    np.testing.assert_allclose(x, exact(0.5), rtol=1e-12)


def test_imex_rk3_handles_a_stiff_linear_term():
    """Crank-Nicolson on L stays stable at h |L| = 1000, where an explicit method would blow up"""
    linear, forcing, x0, exact = linear_problem()
    coarse = utils.integrate_semilinear(linear, constant, 0.0, x0, forcing, 0.5, 1, None, solver=utils.imex_rk3_step)
    fine = utils.integrate_semilinear(linear, constant, 0.0, x0, forcing, 0.5, 64, None, solver=utils.imex_rk3_step)

    assert jnp.all(jnp.isfinite(coarse))
    assert float(jnp.max(jnp.abs(fine - exact(0.5)))) < float(jnp.max(jnp.abs(coarse - exact(0.5)))) + 1e-12
    np.testing.assert_allclose(fine[:2], exact(0.5)[:2], rtol=1e-3)  # the non-stiff components converge


def test_semilinear_rk4_matches_integrate_explicit():
    linear, forcing, x0, _ = linear_problem()
    linear = linear[:2]  # non-stiff part only, plain RK4 is unstable on the rest at this step
    nonlinear = lambda t, x, u, p: jnp.sin(x) + u[:2] * t  # noqa: E731

    semilinear = utils.integrate_semilinear(linear, nonlinear, 0.0, x0[:2], forcing, 0.2, 20, None,
                                            solver=utils.semilinear_rk4_step)
    explicit = utils.integrate_explicit(lambda t, x, u, p: linear * x + nonlinear(t, x, u, p), 0.0, x0[:2], forcing,
                                        0.2, 20, None)
    np.testing.assert_allclose(semilinear, explicit, rtol=1e-14)


def test_complex_linear_operator():
    """Spectral operators can be complex (e.g. advection i k), ETDRK4 then needs the full contour circle"""
    linear = jnp.array((-1.0 + 2.0j, -10.0 - 5.0j))
    forcing = jnp.array((0.5 + 0.1j, -1.0 + 0.0j))
    x0 = jnp.array((1.0 + 0.0j, 0.2 - 0.3j))
    x = utils.integrate_semilinear(linear, constant, 0.0, x0, forcing, 0.3, 1, None, solver=utils.etdrk4_step)

    growth = jnp.exp(linear * 0.3)
    np.testing.assert_allclose(x, growth * x0 + (growth - 1.0) * forcing / linear, rtol=1e-12)


def test_semilinear_jittable():
    linear, forcing, x0, exact = linear_problem()
    step = jax.jit(lambda x: utils.integrate_semilinear(linear, constant, 0.0, x, forcing, 0.5, 1, None,
                                                        solver=utils.etdrk4_step))
    np.testing.assert_allclose(step(x0), exact(0.5), rtol=1e-12)


# # # integrate_implicit
########################################################################################################################
@pytest.mark.parametrize("solver, order", [(utils.backward_euler_step, 1), (utils.implicit_midpoint_step, 2),
                                           (utils.radau_iia3_step, 3)])
def test_implicit_convergence_order(solver, order):
    assert observed_order(utils.integrate_implicit, solver) > order - 0.2


@pytest.mark.parametrize("solver", [utils.backward_euler_step, utils.radau_iia3_step])
def test_l_stable_solvers_damp_stiff_components(solver):
    """
    At h |L| = 1000 one step should shrink the stiff component's distance from its equilibrium -c / L by about
    1 / (h |L|), where implicit midpoint (A-stable, not L-stable) only flips its sign
    """
    linear, forcing, x0, _ = linear_problem()
    equilibrium = -forcing[2] / linear[2]
    x = utils.integrate_implicit(lambda t, x, u, p: linear * x + u, 0.0, x0, forcing, 0.5, 1, None, solver=solver)

    assert jnp.all(jnp.isfinite(x))
    assert float(jnp.abs(x[2] - equilibrium)) < 0.01 * float(jnp.abs(x0[2] - equilibrium))


def test_implicit_midpoint_stays_bounded_on_stiff_components():
    """A-stable but not L-stable, the stiff component flips sign each step without growing"""
    linear, forcing, x0, _ = linear_problem()
    x = utils.integrate_implicit(lambda t, x, u, p: linear * x + u, 0.0, x0, forcing, 5.0, 10, None,
                                 solver=utils.implicit_midpoint_step)
    assert jnp.all(jnp.isfinite(x)) and float(jnp.abs(x[2])) < 10.0


def test_implicit_midpoint_conserves_a_quadratic_energy():
    """Symplectic, so a harmonic oscillator's energy is conserved to the Newton tolerance, unlike RK4's slow drift"""
    oscillator = lambda t, x, u, p: jnp.array((x[1], -x[0]))  # noqa: E731
    x0 = jnp.array((1.0, 0.0))
    midpoint = utils.integrate_implicit(oscillator, 0.0, x0, None, 500.0, 2_000, None,
                                        solver=utils.implicit_midpoint_step)
    rk4 = utils.integrate_explicit(oscillator, 0.0, x0, None, 500.0, 2_000, None)

    assert abs(float(jnp.sum(midpoint ** 2)) - 1.0) < 1e-10
    assert abs(float(jnp.sum(rk4 ** 2)) - 1.0) > 1e-6


def test_radau_handles_stiff_van_der_pol():
    """mu = 1000 relaxation oscillations at a step where explicit RK4 blows up"""
    mu = 1_000.0
    van_der_pol = lambda t, x, u, p: jnp.array((x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]))  # noqa: E731
    x0 = jnp.array((2.0, 0.0))

    implicit = utils.integrate_implicit(van_der_pol, 0.0, x0, None, 100.0, 1_000, None)  # Radau IIA by default
    explicit = utils.integrate_explicit(van_der_pol, 0.0, x0, None, 100.0, 1_000, None)
    assert jnp.all(jnp.isfinite(implicit)) and float(jnp.abs(implicit[0])) < 2.1  # stays on the relaxation cycle
    assert not jnp.all(jnp.isfinite(explicit)) or float(jnp.max(jnp.abs(explicit))) > 1e3


def test_implicit_keeps_the_state_shape_and_is_differentiable():
    x0 = jnp.ones((2, 2))
    decay = lambda t, x, u, p: -x * p  # noqa: E731
    x = utils.integrate_implicit(decay, 0.0, x0, None, 1.0, 4, 2.0)
    assert x.shape == (2, 2)

    grad = jax.grad(lambda rate: jnp.sum(utils.integrate_implicit(decay, 0.0, x0, None, 1.0, 20, rate)))(2.0)
    np.testing.assert_allclose(grad, -4.0 * np.exp(-2.0), rtol=1e-3)  # d/dk of 4 e^{-k} at k = 2


def test_newton_iterations_can_be_set_with_partial():
    solver = functools.partial(utils.radau_iia3_step, newton_iters=2)
    x = utils.integrate_implicit(forced_decay, 0.0, jnp.array((1.0,)), None, 1.0, 10, None, solver=solver)
    assert jnp.all(jnp.isfinite(x))
