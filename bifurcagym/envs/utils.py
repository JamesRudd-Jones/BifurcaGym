import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex


def runge_kutta_4(y0, func, dt):  # TODO sort out standardising which rk4 to use but for now do this
    dt2 = dt / 2.0
    k1 = func(y0, 0)
    k2 = func(y0 + dt2 * k1, dt2)
    k3 = func(y0 + dt2 * k2, dt2)
    k4 = func(y0 + dt * k3, dt)
    y_out = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_out


def rk4_step(f, x: chex.Array, u: chex.Array, dt: float) -> chex.Array:
    """One RK4 step for x' = f(x, u)."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_ode(f, x0: chex.Array, u: chex.Array, dt: float, substeps: int) -> chex.Array:
    """Integrate for dt_total = dt using `substeps` RK4 micro-steps of size dt/substeps."""
    h = dt / float(substeps)

    def body(x, _):
        return rk4_step(f, x, u, h), None

    xf, _ = jax.lax.scan(body, x0, xs=None, length=substeps)
    return xf
