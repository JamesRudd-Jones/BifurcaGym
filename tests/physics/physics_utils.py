"""
Helpers for the physical validity tests.

These are written independently of bifurcagym.envs.utils so that a bug in the env integrators or dynamics cannot
vouch for itself.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jrandom


def rk4_integrate(f: Callable, x0: jnp.ndarray, t0: float, dt: float, n_steps: int) -> jnp.ndarray:
    """Classic RK4 for x' = f(t, x), returns the trajectory of shape (n_steps + 1, ...)"""
    def body(carry, _):
        t, x = carry
        k1 = f(t, x)
        k2 = f(t + dt / 2, x + dt / 2 * k1)
        k3 = f(t + dt / 2, x + dt / 2 * k2)
        k4 = f(t + dt, x + dt * k3)
        x_new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (t + dt, x_new), x_new

    _, xs = jax.lax.scan(body, (t0, x0), None, n_steps)
    return jnp.concatenate([x0[None], xs])


def lagrangian_accel(lagrangian: Callable, q: jnp.ndarray, q_dot: jnp.ndarray, gen_force: jnp.ndarray) -> jnp.ndarray:
    """
    Solves the Euler-Lagrange equations d/dt(dL/dq_dot) - dL/dq = Q for q_ddot:
    M q_ddot = Q + dL/dq - (d^2L / dq_dot dq) q_dot
    """
    mass_matrix = jax.hessian(lagrangian, argnums=1)(q, q_dot)
    dl_dq = jax.grad(lagrangian, argnums=0)(q, q_dot)
    mixed = jax.jacfwd(jax.grad(lagrangian, argnums=1), argnums=0)(q, q_dot) @ q_dot
    return jnp.linalg.solve(mass_matrix, gen_force + dl_dq - mixed)


def rigid_body_energies(com_fn: Callable, omega_fn: Callable, masses: jnp.ndarray, inertias: jnp.ndarray,
                        gravity: float) -> tuple[Callable, Callable]:
    """
    Builds kinetic and potential energy from forward kinematics.

    com_fn(q) -> (n_bodies, 2) centre of mass positions with y pointing up
    omega_fn(q_dot) -> (n_bodies,) angular velocities about each centre of mass
    """
    def kinetic(q, q_dot):
        _, v_com = jax.jvp(com_fn, (q,), (q_dot,))
        omega = omega_fn(q_dot)
        return jnp.sum(0.5 * masses * jnp.sum(v_com ** 2, axis=-1) + 0.5 * inertias * omega ** 2)

    def potential(q):
        return jnp.sum(masses * gravity * com_fn(q)[:, 1])

    return kinetic, potential


def lyapunov_spectrum(step_fn: Callable, x0: jnp.ndarray, n_steps: int, n_transient: int = 1000, d0: float = 1e-7,
                      delta_fn: Callable = None, n_exponents: int = None) -> jnp.ndarray:
    """
    Lyapunov spectrum per step of x_{t+1} = step_fn(x_t) using finite difference tangent vectors with QR
    re-orthonormalisation (Benettin et al. 1980). Divide by dt to get exponents per unit time for flows.
    delta_fn(a, b) gives a - b, e.g. a torus distance for periodic domains.
    n_exponents only tracks the leading exponents, useful for high dimensional systems.
    """
    if delta_fn is None:
        delta_fn = lambda a, b: a - b
    dim = x0.shape[0]
    n_exponents = dim if n_exponents is None else n_exponents

    def burn(x, _):
        return step_fn(x), None

    x0, _ = jax.lax.scan(burn, x0, None, n_transient)

    def body(carry, _):
        x, basis = carry
        x_next = step_fn(x)
        tangent = jax.vmap(lambda v: delta_fn(step_fn(x + d0 * v), x_next) / d0)(basis.T).T
        q, r = jnp.linalg.qr(tangent)
        signs = jnp.sign(jnp.diag(r))
        return (x_next, q * signs), jnp.log(jnp.abs(jnp.diag(r)))

    _, log_growth = jax.lax.scan(body, (x0, jnp.eye(dim, n_exponents, dtype=x0.dtype)), None, n_steps)
    return jnp.sort(jnp.mean(log_growth, axis=0))[::-1]


def random_states(key, n: int, low, high) -> jnp.ndarray:
    low = jnp.asarray(low, dtype=jnp.float64)
    high = jnp.asarray(high, dtype=jnp.float64)
    return jrandom.uniform(key, (n,) + low.shape, minval=low, maxval=high, dtype=jnp.float64)
