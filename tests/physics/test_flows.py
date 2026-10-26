"""
Physical validity of the kinematic flow fields that the swimmer is advected through.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from bifurcagym.envs.continuous_time_chaos import (arnold_beltrami_childress_flow,
                                                   bickley_jet_flow,
                                                   double_gyre_flow,
                                                   quadruple_gyre_flow,
                                                   )
from physics_utils import random_states


pytestmark = pytest.mark.physics

NUM_POINTS = 256


def planar_cases():
    double_gyre = double_gyre_flow.DoubleGyreFlowCSCA()
    quadruple_gyre = quadruple_gyre_flow.QuadrupleGyreFlowCSCA()
    bickley = bickley_jet_flow.BickleyJetFlowCSCA()
    return {"DoubleGyreFlow": double_gyre, "QuadrupleGyreFlow": quadruple_gyre, "BickleyJetFlow": bickley}


def planar_velocity(env, params):
    def velocity(xy, t):
        return jnp.stack(env._get_flow_velocity(xy[0], xy[1], t, params))
    return velocity


def sample_points(env, key):
    key_xy, key_t = jrandom.split(key)
    low = jnp.array((env.x_bounds[0], env.y_bounds[0]))
    high = jnp.array((env.x_bounds[1], env.y_bounds[1]))
    return random_states(key_xy, NUM_POINTS, low, high), random_states(key_t, NUM_POINTS, 0.0, 100.0)


@pytest.mark.parametrize("case", planar_cases().keys())
def test_planar_flow_is_incompressible(case):
    env = planar_cases()[case]
    velocity = planar_velocity(env, env.default_params)
    xys, ts = sample_points(env, jrandom.key(0))

    divergence = jax.vmap(lambda xy, t: jnp.trace(jax.jacfwd(velocity)(xy, t)))(xys, ts)
    np.testing.assert_allclose(divergence, 0.0, atol=1e-12)


@pytest.mark.parametrize("case", ["DoubleGyreFlow", "QuadrupleGyreFlow"])
def test_gyre_walls_have_no_normal_flow(case):
    """The gyres live in a closed box, so the velocity normal to every wall must vanish"""
    env = planar_cases()[case]
    velocity = planar_velocity(env, env.default_params)
    s = jnp.linspace(0.0, 1.0, 101)
    ts = jnp.linspace(0.0, 50.0, 101)
    x_lo, x_hi = env.x_bounds
    y_lo, y_hi = env.y_bounds
    xs = x_lo + s * (x_hi - x_lo)
    ys = y_lo + s * (y_hi - y_lo)

    u_left = jax.vmap(lambda y, t: velocity(jnp.array((x_lo, y)), t)[0])(ys, ts)
    u_right = jax.vmap(lambda y, t: velocity(jnp.array((x_hi, y)), t)[0])(ys, ts)
    v_bottom = jax.vmap(lambda x, t: velocity(jnp.array((x, y_lo)), t)[1])(xs, ts)
    v_top = jax.vmap(lambda x, t: velocity(jnp.array((x, y_hi)), t)[1])(xs, ts)

    for wall in (u_left, u_right, v_bottom, v_top):
        np.testing.assert_allclose(wall, 0.0, atol=1e-12)


def test_double_gyre_matches_shadden_closed_form():
    """Shadden, Lekien and Marsden (2005), psi = A sin(pi f(x, t)) sin(pi y)"""
    env = double_gyre_flow.DoubleGyreFlowCSCA()
    params = env.default_params
    xys, ts = sample_points(env, jrandom.key(1))

    def psi(xy, t):
        a = params.epsilon * jnp.sin(params.omega * t)
        b = 1 - 2 * params.epsilon * jnp.sin(params.omega * t)
        f = a * xy[0] ** 2 + b * xy[0]
        return params.A * jnp.sin(jnp.pi * f) * jnp.sin(jnp.pi * xy[1])

    def expected(xy, t):
        grad = jax.grad(psi)(xy, t)
        return jnp.array((-grad[1], grad[0]))  # u = -dpsi/dy, v = dpsi/dx

    np.testing.assert_allclose(jax.vmap(planar_velocity(env, params))(xys, ts),
                               jax.vmap(expected)(xys, ts), rtol=1e-10, atol=1e-12)


def test_bickley_jet_matches_rypina_streamfunction():
    """Rypina et al. (2007), psi = -U0 L tanh(y / L) + U0 L sech^2(y / L) sum_n eps_n cos(k_n (x - c_n t))"""
    env = bickley_jet_flow.BickleyJetFlowCSCA()
    params = env.default_params
    xys, ts = sample_points(env, jrandom.key(2))

    def psi(xy, t):
        y_scaled = xy[1] / params.L
        waves = jnp.sum(params.epsilon * jnp.cos(params.k * (xy[0] - params.c * t)))
        return -params.U0 * params.L * jnp.tanh(y_scaled) + params.U0 * params.L * waves / jnp.cosh(y_scaled) ** 2

    def expected(xy, t):
        grad = jax.grad(psi)(xy, t)
        return jnp.array((-grad[1], grad[0]))

    np.testing.assert_allclose(jax.vmap(planar_velocity(env, params))(xys, ts),
                               jax.vmap(expected)(xys, ts), rtol=1e-10, atol=1e-12)


def test_bickley_jet_is_periodic_over_the_wrapped_domain():
    """The env wraps x modulo x_len, so the flow must be continuous across that seam"""
    env = bickley_jet_flow.BickleyJetFlowCSCA()
    velocity = planar_velocity(env, env.default_params)
    ys = jnp.linspace(env.y_bounds[0], env.y_bounds[1], 51)
    ts = jnp.linspace(0.0, 50.0, 51)

    at_start = jax.vmap(lambda y, t: velocity(jnp.array((env.x_bounds[0], y)), t))(ys, ts)
    at_end = jax.vmap(lambda y, t: velocity(jnp.array((env.x_bounds[0] + env.x_len, y)), t))(ys, ts)
    np.testing.assert_allclose(at_start, at_end, atol=1e-10)


def abc_velocity(env, params):
    def velocity(xyz):
        return jnp.stack(env._get_flow_velocity(xyz[0], xyz[1], xyz[2], params))
    return velocity


def abc_points():
    return random_states(jrandom.key(3), NUM_POINTS, jnp.zeros(3), 2 * jnp.pi * jnp.ones(3))


def test_abc_flow_is_incompressible_and_beltrami():
    """ABC flow is an exact steady Euler solution with curl u = u"""
    env = arnold_beltrami_childress_flow.ABCFlowCSCA()
    velocity = abc_velocity(env, env.default_params)
    points = abc_points()

    jac = jax.vmap(jax.jacfwd(velocity))(points)
    divergence = jnp.trace(jac, axis1=1, axis2=2)
    curl = jnp.stack((jac[:, 2, 1] - jac[:, 1, 2],
                      jac[:, 0, 2] - jac[:, 2, 0],
                      jac[:, 1, 0] - jac[:, 0, 1]), axis=-1)

    np.testing.assert_allclose(divergence, 0.0, atol=1e-12)
    np.testing.assert_allclose(curl, jax.vmap(velocity)(points), atol=1e-12)


def test_abc_flow_is_periodic_over_the_wrapped_domain():
    env = arnold_beltrami_childress_flow.ABCFlowCSCA()
    velocity = abc_velocity(env, env.default_params)
    points = abc_points()

    for axis in range(3):
        shifted = points.at[:, axis].add(env.L)
        np.testing.assert_allclose(jax.vmap(velocity)(shifted), jax.vmap(velocity)(points), atol=1e-12)
