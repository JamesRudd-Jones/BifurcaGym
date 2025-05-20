import jax
import jax.numpy as jnp
import jax.random as jrandom


def runge_kutta_4(y0, func, dt):
    dt2 = dt / 2.0
    k1 = func(y0, 0)
    k2 = func(y0 + dt2 * k1, dt2)
    k3 = func(y0 + dt2 * k2, dt2)
    k4 = func(y0 + dt * k3, dt)
    y_out = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_out