import jax
import jax.numpy as jnp
import jax.random as jrandom


def _runge_kutta_update(runner, unused):
    u_K, ind = runner
    dt = self.dt / (3 - ind)
    u_K = u_save_K + dt * self.nlterm(u_K, f_K)
    u_K = (u_K + 0.5 * self.lin_K * dt * u_save_K) / (1. - 0.5 * self.lin_K * dt)

    ind += 1

    return (u_K, ind), None


final_runner_state = jax.lax.scan(_runge_kutta_update, (u_K, 0), None, 3)


def runge_kutta_4(y0, func, dt):
    dt2 = dt / 2.0
    k1 = func(y0, 0)
    k2 = func(y0 + dt2 * k1, dt2)
    k3 = func(y0 + dt2 * k2, dt2)
    k4 = func(y0 + dt * k3, dt)
    y_out = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_out