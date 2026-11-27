import jax
import jax.numpy as jnp
import chex
from typing import Callable


def _integrate(one_step: Callable, t0: float, x0: chex.Array, dt: float, substeps: int) -> chex.Array:
    """`substeps` steps of size dt / substeps of one_step(t, x, h) -> x_next"""
    h = dt / float(substeps)

    def body(carry, _):
        t, x = carry
        return (t + h, one_step(t, x, h)), None

    (_, xf), _ = jax.lax.scan(body, (t0, x0), xs=None, length=substeps)

    return xf


# # # Explicit solvers, step(f, t, x, u, h, params)
########################################################################################################################
def euler_step(f: Callable, t: float, x: chex.Array, u: chex.Array, h: float, params) -> chex.Array:
    """One forward Euler step for x' = f(t, x, u), first order"""
    return x + h * f(t, x, u, params)


def rk4_step(f: Callable, t: float, x: chex.Array, u: chex.Array, h: float, params) -> chex.Array:
    """One classic RK4 step for x' = f(t, x, u), fourth order"""
    k1 = f(t, x, u, params)
    k2 = f(t + 0.5 * h, x + 0.5 * h * k1, u, params)
    k3 = f(t + 0.5 * h, x + 0.5 * h * k2, u, params)
    k4 = f(t + h, x + h * k3, u, params)

    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_explicit(f: Callable, t0: float, x0: chex.Array, u: chex.Array, dt: float, substeps: int, params,
                       solver: Callable = rk4_step) -> chex.Array:
    return _integrate(lambda t, x, h: solver(f, t, x, u, h, params), t0, x0, dt, substeps)


# # # Implicit solvers, step(f, t, x, u, h, params, newton_iters=10), use functools.partial to change newton_iters
########################################################################################################################
# Two stage Radau IIA, c = (1/3, 1), stiffly accurate so the step's result is the last stage
_RADAU_IIA3_A = ((5.0 / 12.0, -1.0 / 12.0), (3.0 / 4.0, 1.0 / 4.0))
_RADAU_IIA3_C = (1.0 / 3.0, 1.0)


def _newton(residual: Callable, z0: chex.Array, iters: int) -> chex.Array:
    def body(z, _):
        return z - jnp.linalg.solve(jax.jacfwd(residual)(z), residual(z)), None

    z, _ = jax.lax.scan(body, z0, None, length=iters)

    return z


def _flat_rhs(f: Callable, shape, u: chex.Array, params) -> Callable:
    """f on flattened states, Newton works on vectors"""
    return lambda t, x_flat: jnp.ravel(f(t, jnp.reshape(x_flat, shape), u, params))


def backward_euler_step(f: Callable, t: float, x: chex.Array, u: chex.Array, h: float, params,
                        newton_iters: int = 10) -> chex.Array:
    rhs, x_flat = _flat_rhs(f, jnp.shape(x), u, params), jnp.ravel(x)
    new = _newton(lambda y: y - x_flat - h * rhs(t + h, y), x_flat, newton_iters)

    return jnp.reshape(new, jnp.shape(x))


def implicit_midpoint_step(f: Callable, t: float, x: chex.Array, u: chex.Array, h: float, params,
                           newton_iters: int = 10) -> chex.Array:
    """
    One implicit midpoint step, second order, A-stable and symplectic, so energy stays bounded over long runs of
    conservative systems (pendulums, three body), but stiff components are not damped
    """
    rhs, x_flat = _flat_rhs(f, jnp.shape(x), u, params), jnp.ravel(x)
    new = _newton(lambda y: y - x_flat - h * rhs(t + h / 2, (x_flat + y) / 2), x_flat, newton_iters)

    return jnp.reshape(new, jnp.shape(x))


def radau_iia3_step(f: Callable, t: float, x: chex.Array, u: chex.Array, h: float, params,
                    newton_iters: int = 10) -> chex.Array:
    rhs, x_flat = _flat_rhs(f, jnp.shape(x), u, params), jnp.ravel(x)
    n = x_flat.shape[0]
    a = jnp.asarray(_RADAU_IIA3_A, dtype=x_flat.dtype)

    def residual(z):
        stages = jnp.reshape(z, (2, n))
        slopes = jnp.stack([rhs(t + c * h, x_flat + stage) for c, stage in zip(_RADAU_IIA3_C, stages, strict=True)])

        return jnp.ravel(stages - h * a @ slopes)

    stages = jnp.reshape(_newton(residual, jnp.zeros(2 * n, dtype=x_flat.dtype), newton_iters), (2, n))

    return jnp.reshape(x_flat + stages[1], jnp.shape(x))


def integrate_implicit(f: Callable, t0: float, x0: chex.Array, u: chex.Array, dt: float, substeps: int, params,
                       solver: Callable = radau_iia3_step) -> chex.Array:
    return _integrate(lambda t, x, h: solver(f, t, x, u, h, params), t0, x0, dt, substeps)


# # # Semilinear solvers, step(linear, nonlinear, t, x, u, h, params) for x' = L x + N(t, x, u) with diagonal L
########################################################################################################################
def _etdrk4_coefficients(linear: chex.Array, h: float, n_contour: int = 32):
    lh = h * linear
    real = not jnp.iscomplexobj(linear)
    # A real operator only needs the upper half circle (the real part of its mean equals the full circle's), a complex
    # one needs the full circle
    angles = (jnp.arange(1, n_contour + 1) - 0.5) / n_contour * (jnp.pi if real else 2.0 * jnp.pi)
    roots = jnp.exp(1j * angles)
    lr = lh[..., None] + roots

    def mean(values):
        values = jnp.mean(values, axis=-1)

        return jnp.real(values) if real else values

    e = jnp.exp(lh)
    e2 = jnp.exp(lh / 2)
    q = h * mean((jnp.exp(lr / 2) - 1) / lr)
    f1 = h * mean((-4 - lr + jnp.exp(lr) * (4 - 3 * lr + lr ** 2)) / lr ** 3)
    f2 = h * mean((2 + lr + jnp.exp(lr) * (-2 + lr)) / lr ** 3)
    f3 = h * mean((-4 - 3 * lr - lr ** 2 + jnp.exp(lr) * (4 - lr)) / lr ** 3)

    return e, e2, q, f1, f2, f3


def imex_rk3_step(linear: chex.Array, nonlinear: Callable, t: float, x: chex.Array, u: chex.Array, h: float,
                  params) -> chex.Array:
    def stage(x_stage, stage_index):
        h_stage = h / (3 - stage_index)
        x_stage = x + h_stage * nonlinear(t, x_stage, u, params)

        return (x_stage + 0.5 * linear * h_stage * x) / (1.0 - 0.5 * linear * h_stage), None

    x_next, _ = jax.lax.scan(stage, x, jnp.arange(3))

    return x_next


def etdrk4_step(linear: chex.Array, nonlinear: Callable, t: float, x: chex.Array, u: chex.Array, h: float,
                params) -> chex.Array:
    e, e2, q, f1, f2, f3 = _etdrk4_coefficients(linear, h)
    n_x = nonlinear(t, x, u, params)
    a = e2 * x + q * n_x
    n_a = nonlinear(t + h / 2, a, u, params)
    b = e2 * x + q * n_a
    n_b = nonlinear(t + h / 2, b, u, params)
    c = e2 * a + q * (2 * n_b - n_x)
    n_c = nonlinear(t + h, c, u, params)

    return e * x + n_x * f1 + 2 * (n_a + n_b) * f2 + n_c * f3


def semilinear_rk4_step(linear: chex.Array, nonlinear: Callable, t: float, x: chex.Array, u: chex.Array, h: float,
                        params) -> chex.Array:
    return rk4_step(lambda t_, x_, u_, p_: linear * x_ + nonlinear(t_, x_, u_, p_), t, x, u, h, params)


def integrate_semilinear(linear: chex.Array, nonlinear: Callable, t0: float, x0: chex.Array, u: chex.Array, dt: float,
                         substeps: int, params, solver: Callable = imex_rk3_step) -> chex.Array:
    return _integrate(lambda t, x, h: solver(linear, nonlinear, t, x, u, h, params), t0, x0, dt, substeps)


# # # Other utils
########################################################################################################################
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


def action_permutations_generation(num_actions) -> chex.Array:
    # # 7 actions: noop and rook moves, no diagonals for scalability as of now
    # self.action_array: chex.Array = jnp.array(((0.0, 0.0, 0.0),
    #                                            (1.0, 0.0, 0.0),
    #                                            (-1.0, 0.0, 0.0),
    #                                            (0.0, 1.0, 0.0),
    #                                            (0.0, -1.0, 0.0),
    #                                            (0.0, 0.0, 1.0),
    #                                            (0.0, 0.0, -1.0),
    #                                            ))

    action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))
    idx = jnp.arange(action_array.shape[0] ** num_actions)
    powers = action_array.shape[0] ** jnp.arange(num_actions)
    digits = (idx[:, None] // powers[None, :]) % action_array.shape[0]

    return action_array[digits]
    # TODO this does not scale very nicely
