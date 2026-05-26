import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex
from typing import Callable


def rk4_step(f, x: chex.Array, u: chex.Array, dt: float, params) -> chex.Array:
    """One RK4 step for x' = f(x, u)."""
    k1 = f(x, u, params)
    k2 = f(x + 0.5 * dt * k1, u, params)
    k3 = f(x + 0.5 * dt * k2, u, params)
    k4 = f(x + dt * k3, u, params)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_ode(f, x0: chex.Array, u: chex.Array, dt: float, substeps: int, params) -> chex.Array:
    """Integrate for dt_total = dt using `substeps` RK4 micro-steps of size dt/substeps."""
    h = dt / float(substeps)

    def body(x, _):
        return rk4_step(f, x, u, h, params), None

    xf, _ = jax.lax.scan(body, x0, xs=None, length=substeps)
    return xf


def newton_solve_2d(F: Callable[[chex.Array], chex.Array], x0: chex.Array, iters: int = 40, tol: float = 1e-10,
                    damping: float = 1.0) -> chex.Array:
    """
    Solve F(x)=0 for x in R^2 using Newton's method with autodiff Jacobian.
    """

    def step(x):
        fx = F(x)
        J = jax.jacobian(F)(x)  # 2x2
        # Solve J dx = -f
        dx = jnp.linalg.solve(J, -fx)
        x_new = x + damping * dx
        return x_new, fx

    # x = x0
    #
    # for _ in range(iters):
    #     x_new, fx = step(x)
    #     if float(jnp.linalg.norm(fx)) < tol:
    #         x = x_new
    #         break
    #     x = x_new

    def _step_func(x, unused):
        x_new, fx = step(x)
        r = jnp.linalg.norm(fx)
        return x_new, (x_new, r)

    _, (x_vals, r_vals) = jax.lax.scan(_step_func, x0, None, iters)

    def select_first_converged(xs, rs, tol):
        hits = rs < tol
        idxs = jnp.arange(rs.shape[0])
        # choose first hit; if no hit, choose last index
        first = jnp.min(jnp.where(hits, idxs, rs.shape[0] - 1))
        return xs[first], first, hits[first]

    x_star, k_star, ok = select_first_converged(x_vals, r_vals, tol)

    return x_star
