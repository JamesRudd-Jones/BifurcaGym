from collections.abc import Callable
import dataclasses
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from bifurcagym import metadata
from bifurcagym import spaces


_WRAPPER_ATTRS = ("_wrapped_metrics_env",
                  "_wrapped_autoreset_env",
                  "_wrapped_normalised_env",
                  "_wrapped_periodic_env",
                  )

_MAX_DEFAULT_EXPONENTS = 512  # limit to prevent too much computational load


def unwrap(env):
    while True:
        inner = next((vars(env)[attr] for attr in _WRAPPER_ATTRS if attr in vars(env)), None)

        if inner is None:
            return env

        env = inner


def step_duration(env, params) -> float | None:
    """
    For continuous time environments, if it is BifurcaGym's it is .dt but HydroGym uses .action_time
    """
    base = unwrap(env)

    if hasattr(base, "dt"):
        return float(base.dt)

    if hasattr(params, "action_time"):
        return float(params.action_time)

    return None


def max_steps_in_ep(env, params) -> int | None:
    if hasattr(params, "max_steps_in_ep"):
        return int(params.max_steps_in_ep)

    base = unwrap(env)

    return int(base.max_steps_in_ep) if hasattr(base, "max_steps_in_ep") else None


def zero_action(env, params) -> chex.Array:
    space = unwrap(env).action_space(params)

    if isinstance(space, spaces.Discrete):
        return jnp.zeros((1,), dtype=space.dtype)

    low = jnp.broadcast_to(jnp.asarray(space.low), space.shape)
    high = jnp.broadcast_to(jnp.asarray(space.high), space.shape)

    return jnp.clip(jnp.zeros(space.shape, dtype=space.dtype), low, high).astype(space.dtype)


class _StateVector:
    """
    Converts between an env state and a real vector of its float leaves. The integer time counter and any PRNG keys
    are left alone, and complex leaves (HydroGym's spectral fields) become their real and imaginary parts, so the
    tangent space is real and QR and the exponents behave as usual.
    """

    def __init__(self, example_state):
        leaves, self.treedef = jax.tree.flatten(example_state)
        self.kinds = []  # None for leaves left alone, else (shape, dtype, is_complex)

        for leaf in leaves:
            dtype = jnp.result_type(leaf)

            if jnp.issubdtype(dtype, jnp.inexact):
                self.kinds.append((jnp.shape(leaf), dtype, bool(jnp.issubdtype(dtype, jnp.complexfloating))))

            else:
                self.kinds.append(None)

        self.dim = sum(int(np.prod(kind[0])) * (2 if kind[2] else 1) for kind in self.kinds if kind is not None)

    def to_vec(self, state) -> chex.Array:
        parts = []

        for leaf, kind in zip(jax.tree.leaves(state), self.kinds, strict=True):

            if kind is None:
                continue

            flat = jnp.ravel(leaf)
            parts.extend((jnp.real(flat), jnp.imag(flat)) if kind[2] else (flat,))

        return jnp.concatenate(parts)

    def from_vec(self, vec: chex.Array, like_state):
        new_leaves, i = [], 0

        for leaf, kind in zip(jax.tree.leaves(like_state), self.kinds, strict=True):

            if kind is None:
                new_leaves.append(leaf)
                continue

            shape, dtype, is_complex = kind
            size = int(np.prod(shape))

            if is_complex:
                value = vec[i:i + size] + 1j * vec[i + size:i + 2 * size]
                i += 2 * size

            else:
                value = vec[i:i + size]
                i += size

            new_leaves.append(jnp.reshape(value, shape).astype(dtype))

        return self.treedef.unflatten(new_leaves)


def _action_fn(env, params, action: chex.Array | None, policy: Callable | None) -> Callable:
    base = unwrap(env)

    if policy is not None:
        return lambda state, key: policy(base.get_obs(state), key)

    fixed = zero_action(env, params) if action is None else jnp.asarray(action)

    return lambda state, key: fixed


class LyapunovSpectrum(NamedTuple):
    per_step: chex.Array  # exponents per env step, largest first
    per_time: chex.Array | None  # per unit time, None when the env has no time unit (maps)
    stderr: chex.Array  # standard error of per_step from batch means over the run
    step_duration: float | None
    n_steps: int
    finite: bool  # False if the orbit escaped to infinity thus making the exponents meaningless


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8))
def _benettin(base, params, state, action_fn, n_steps: int, n_transient: int, n_exponents: int, method: str,
              d0: float, key):
    converter = _StateVector(state)

    def step(state, key):
        action_key, step_key = jrandom.split(key)
        return base.step_env(action_fn(state, action_key), state, params, step_key)[2]

    def burn(state, key):
        return step(state, key), None

    burn_key, run_key = jrandom.split(key)
    state, _ = jax.lax.scan(burn, state, jrandom.split(burn_key, n_transient))

    def body(carry, key):
        state, basis = carry
        x = converter.to_vec(state)

        def f(vec):
            new_state = step(converter.from_vec(vec, state), key)
            return converter.to_vec(new_state), new_state

        if method == "jvp":
            x_next, f_lin, new_state = jax.linearize(f, x, has_aux=True)
            tangent = jax.vmap(f_lin, in_axes=1, out_axes=1)(basis)

        else:
            x_next, new_state = f(x)
            tangent = jax.vmap(lambda v: (f(x + d0 * v)[0] - x_next) / d0, in_axes=1, out_axes=1)(basis)

        q, r = jnp.linalg.qr(tangent)
        signs = jnp.where(jnp.diag(r) < 0, -1.0, 1.0).astype(q.dtype)

        return (new_state, q * signs), jnp.log(jnp.abs(jnp.diag(r)))

    basis = jnp.eye(converter.dim, n_exponents, dtype=converter.to_vec(state).dtype)
    (final_state, _), log_growth = jax.lax.scan(body, (state, basis), jrandom.split(run_key, n_steps))

    # A tangent collapsing to exactly zero (e.g. a reset or clip inside the step) gives log 0 = -inf, which is strong
    # contraction rather than a failure, so only a non-finite state or NaN growth counts
    finite = ~jnp.any(jnp.isnan(log_growth)) & jnp.all(jnp.isfinite(converter.to_vec(final_state)))

    return log_growth, finite


def lyapunov_spectrum(env,
                      params,
                      state,
                      *,
                      action: chex.Array | None = None,
                      policy: Callable | None = None,
                      n_steps: int = 10_000,
                      n_transient: int = 1_000,
                      n_exponents: int | None = None,
                      method: str = "jvp",
                      d0: float = 1e-8,
                      key: chex.PRNGKey | None = None,
                      n_batches: int = 10,
                      ) -> LyapunovSpectrum:
    """
    Lyapunov spectrum of the env's dynamics by the Benettin et al. (1980) method: tangent vectors are pushed through
    each step and re-orthonormalised by QR, the exponents are the mean log growth along each direction
    """
    base = unwrap(env)
    dim = _StateVector(state).dim

    if n_exponents is None:
        if dim > _MAX_DEFAULT_EXPONENTS:
            raise ValueError(f"{base.name} has a {dim} dimensional state, pass n_exponents to track only the leading "
                             f"exponents")

        n_exponents = dim

    n_exponents = min(n_exponents, dim)
    key = jrandom.key(0) if key is None else key

    log_growth, finite = _benettin(base, params, state, _action_fn(env, params, action, policy), n_steps, n_transient,
                                   n_exponents, method, d0, key)

    return _spectrum_from_growth(log_growth, finite, n_batches, step_duration(env, params))


def _spectrum_from_growth(log_growth: chex.Array, finite, n_batches: int, duration: float | None) -> LyapunovSpectrum:
    """Exponents (largest first) and their batch means standard errors from the per step log growth"""
    n_steps, n_exponents = log_growth.shape
    per_step = jnp.mean(log_growth, axis=0)
    order = jnp.argsort(per_step)[::-1]
    per_step = per_step[order]
    batch_means = jnp.mean(jnp.reshape(log_growth[:n_steps - n_steps % n_batches], (n_batches, -1, n_exponents)),
                           axis=1)[:, order]
    stderr = jnp.std(batch_means, axis=0, ddof=1) / np.sqrt(n_batches)

    return LyapunovSpectrum(per_step=per_step,
                            per_time=None if duration is None else per_step / duration,
                            stderr=stderr,
                            step_duration=duration,
                            n_steps=n_steps,
                            finite=bool(finite),
                            )


def lyapunov_time(spectrum: LyapunovSpectrum) -> tuple[float, float | None]:
    leading = float(spectrum.per_step[0])

    if leading <= 0.0:
        return np.inf, (None if spectrum.step_duration is None else np.inf)

    steps = 1.0 / leading

    return steps, (None if spectrum.step_duration is None else steps * spectrum.step_duration)


def episode_efoldings(env, params, spectrum: LyapunovSpectrum) -> float | None:
    """
    How many Lyapunov times fit in an episode, max_steps_in_ep * lambda_1. This calcs how far beyond the predictable
    horizon an agent has to act. Negative when the dynamics contract
    """
    steps = max_steps_in_ep(env, params)

    return None if steps is None else steps * float(spectrum.per_step[0])


def n_positive_exponents(spectrum: LyapunovSpectrum, n_stderr: float = 3.0) -> int:
    """
    Exponents positive beyond n_stderr standard errors, more than one means hyperchaos
    """
    return int(jnp.sum(spectrum.per_step > n_stderr * spectrum.stderr))


def kaplan_yorke_dimension(exponents: chex.Array) -> float:
    exponents = np.sort(np.asarray(exponents))[::-1]
    partial_sums = np.cumsum(exponents)

    if partial_sums[0] < 0:
        return 0.0

    k = int(np.max(np.nonzero(partial_sums >= 0)[0])) + 1

    if k == len(exponents):
        return float(k)

    return float(k + partial_sums[k - 1] / abs(exponents[k]))


class HyperbolicityCheck(NamedTuple):
    verdict: str  # "hyperbolic", "non-hyperbolic", "inconclusive" or "not applicable"
    spectrum: LyapunovSpectrum
    n_unstable: int
    n_neutral: int
    n_stable: int
    angles: np.ndarray | None
    min_angle: float | None
    low_angle: float | None  # the `quantile` quantile of the angles
    min_expansion: float | None  # 1D maps: smallest per step log|f'|, uniformly expanding if positive
    evidence: str


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
def _forward_qr(base, params, state, action_fn, n_steps: int, n_transient: int, n_exponents: int, key):
    """Benettin's forward pass, also keeping each step's orthonormal basis Q and triangular factor R for the CLVs"""
    converter = _StateVector(state)

    def step(state, key):
        action_key, step_key = jrandom.split(key)
        return base.step_env(action_fn(state, action_key), state, params, step_key)[2]

    def burn(state, key):
        return step(state, key), None

    burn_key, run_key = jrandom.split(key)
    state, _ = jax.lax.scan(burn, state, jrandom.split(burn_key, n_transient))

    def body(carry, key):
        state, basis = carry
        x = converter.to_vec(state)

        def f(vec):
            new_state = step(converter.from_vec(vec, state), key)
            return converter.to_vec(new_state), new_state

        _, f_lin, new_state = jax.linearize(f, x, has_aux=True)
        q, r = jnp.linalg.qr(jax.vmap(f_lin, in_axes=1, out_axes=1)(basis))
        signs = jnp.where(jnp.diag(r) < 0, -1.0, 1.0).astype(q.dtype)
        q, r = q * signs, r * signs[:, None]  # still q r = J basis, now with a positive diagonal

        return (new_state, q), (jnp.log(jnp.abs(jnp.diag(r))), q, r)

    basis = jnp.eye(converter.dim, n_exponents, dtype=converter.to_vec(state).dtype)
    (final_state, _), (log_growth, qs, rs) = jax.lax.scan(body, (state, basis), jrandom.split(run_key, n_steps))
    finite = ~jnp.any(jnp.isnan(log_growth)) & jnp.all(jnp.isfinite(converter.to_vec(final_state)))

    return log_growth, qs, rs, basis, finite


@partial(jax.jit, static_argnums=(3,))
def _clv_angles(qs: chex.Array, rs: chex.Array, basis: chex.Array, n_split: int, key) -> chex.Array:
    q_before = jnp.concatenate((basis[None], qs[:-1]), axis=0)  # Q_n, the basis entering step n
    k = rs.shape[-1]
    c_last = jnp.triu(jrandom.uniform(key, (k, k), minval=0.5, maxval=1.0, dtype=rs.dtype))

    def backward(c, inputs):
        q_n, r_next = inputs
        c = jax.scipy.linalg.solve_triangular(r_next, c, lower=False)
        c = c / jnp.linalg.norm(c, axis=0)
        clvs = q_n @ c
        first, _ = jnp.linalg.qr(clvs[:, :n_split])
        rest, _ = jnp.linalg.qr(clvs[:, n_split:])
        cosine = jnp.max(jnp.linalg.svd(first.T @ rest, compute_uv=False))

        return c, jnp.arccos(jnp.clip(cosine, 0.0, 1.0))

    _, angles = jax.lax.scan(backward, c_last / jnp.linalg.norm(c_last, axis=0), (q_before, rs), reverse=True)

    return angles


def hyperbolicity(env,
                  params,
                  state,
                  *,
                  action: chex.Array | None = None,
                  n_steps: int = 20_000,
                  n_transient: int = 1_000,
                  n_converge: int = 1_000,
                  n_stderr: float = 3.0,
                  rtol: float = 0.01,
                  tangency_angle: float = 0.01,
                  hyperbolic_angle: float = 0.1,
                  quantile: float = 0.001,
                  key: chex.PRNGKey | None = None,
                  n_batches: int = 10,
                  ) -> HyperbolicityCheck:
    base = unwrap(env)
    dim = _StateVector(state).dim

    if dim > _MAX_DEFAULT_EXPONENTS:
        raise ValueError(f"{base.name} has a {dim} dimensional state, too large for a full CLV computation")

    forward_key, backward_key = jrandom.split(jrandom.key(0) if key is None else key)
    log_growth, qs, rs, basis, finite = _forward_qr(base, params, state, _action_fn(env, params, action, None),
                                                    n_steps, n_transient, dim, forward_key)
    spectrum = _spectrum_from_growth(log_growth, finite, n_batches, step_duration(env, params))

    growth = np.asarray(log_growth)
    column_means = growth.mean(axis=0)
    batches = growth[:n_steps - n_steps % n_batches].reshape(n_batches, -1, dim).mean(axis=1)
    column_stderr = batches.std(axis=0, ddof=1) / np.sqrt(n_batches)
    tolerance = np.maximum(n_stderr * column_stderr, rtol * float(np.max(np.abs(column_means))))
    n_unstable = int(np.sum(column_means > tolerance))
    n_neutral = int(np.sum(np.abs(column_means) <= tolerance))
    n_stable = dim - n_unstable - n_neutral
    counts = {"n_unstable": n_unstable, "n_neutral": n_neutral, "n_stable": n_stable}
    empty = {"angles": None, "min_angle": None, "low_angle": None, "min_expansion": None}

    if not spectrum.finite:
        return HyperbolicityCheck("not applicable", spectrum, **counts, **empty, evidence="the orbit blew up")

    if n_unstable == 0:
        return HyperbolicityCheck("not applicable", spectrum, **counts, **empty,
                                  evidence="no expanding direction, so the orbit is not chaotic")

    if n_stable == 0:
        min_expansion = float(jnp.min(log_growth[:, :n_unstable]))
        verdict = "hyperbolic" if min_expansion > 0 else "non-hyperbolic"
        reading = "positive, uniformly expanding" if min_expansion > 0 else "not positive, e.g. a critical point"
        evidence = (f"no stable direction, so the test is uniform expansion: the smallest per step log stretching is "
                    f"{min_expansion:.3g} ({reading})")

        return HyperbolicityCheck(verdict, spectrum, **counts, **{**empty, "min_expansion": min_expansion},
                                  evidence=evidence)

    if n_steps <= 2 * n_converge:
        raise ValueError(f"n_steps ({n_steps}) must exceed 2 * n_converge ({n_converge})")

    angles = np.asarray(_clv_angles(qs, rs, basis, n_unstable + n_neutral, backward_key))[n_converge:-n_converge]
    min_angle = float(np.min(angles))
    low_angle = float(np.quantile(angles, quantile))

    if low_angle < tangency_angle:
        verdict, reading = "non-hyperbolic", "collapsing towards zero"

    elif low_angle > hyperbolic_angle:
        verdict, reading = "hyperbolic", "bounded away from zero"

    else:
        verdict, reading = "inconclusive", "small but not clearly zero, re-run longer to see if it keeps falling"

    evidence = (f"angle between the centre-unstable ({n_unstable} + {n_neutral} neutral) and stable ({n_stable}) "
                f"directions: {quantile:.1%} quantile {low_angle:.3g} rad, minimum {min_angle:.3g} rad, {reading}")

    if n_neutral:
        evidence += " (neutral directions grouped with unstable, a dominated splitting test)"

    return HyperbolicityCheck(verdict, spectrum, **counts, angles=angles, min_angle=min_angle, low_angle=low_angle,
                              min_expansion=None, evidence=evidence)


class DynamicsClassification(NamedTuple):
    verdict: str  # see classify_dynamics
    spectrum: LyapunovSpectrum
    n_positive: int
    n_zero: int
    bounded: bool
    period: int | None  # for maps, the period of the attracting cycle when all exponents are negative
    conservative: bool  # exponents sum to ~0 (Hamiltonian or volume preserving), only judged from a full spectrum
    full_spectrum: bool  # False if only the leading exponents were tracked, so negatives beyond them are unknown
    evidence: str  # the verbose reasoning behind the decision


# States above this size keep only random projections of the tail for memory reasons
_MAX_FULL_TAIL_DIM = 4_096
_N_PROJECTIONS = 8


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
def _orbit_summary(base, params, state, action_fn, n_transient: int, n_steps: int, n_tail: int, key):
    converter = _StateVector(state)
    projection_key, burn_key, run_key = jrandom.split(key, 3)
    dtype = converter.to_vec(state).dtype
    projection = jrandom.normal(projection_key, (converter.dim, _N_PROJECTIONS), dtype=dtype) / np.sqrt(converter.dim)
    full_tail = converter.dim <= _MAX_FULL_TAIL_DIM

    def advance(state, key):
        action_key, step_key = jrandom.split(key)
        return base.step_env(action_fn(state, action_key), state, params, step_key)[2]

    def burn(state, key):
        return advance(state, key), None

    def run(state, key):
        new_state = advance(state, key)
        vec = converter.to_vec(new_state)
        return new_state, (jnp.max(jnp.abs(vec)), vec @ projection)

    def tail(state, key):
        new_state = advance(state, key)
        vec = converter.to_vec(new_state)
        return new_state, (vec if full_tail else vec @ projection)

    state, _ = jax.lax.scan(burn, state, jrandom.split(burn_key, n_transient))
    state, (sizes, projected) = jax.lax.scan(run, state, jrandom.split(run_key, n_steps - n_tail))
    _, tail_states = jax.lax.scan(tail, state, jrandom.split(jrandom.fold_in(run_key, 1), n_tail))
    half = (n_steps - n_tail) // 2

    return jnp.max(sizes[:half]), jnp.max(sizes[half:]), projected, tail_states


def _recurrent(projected: np.ndarray, rtol: float = 0.02, exclude: float = 0.1) -> bool:
    """
    Calc for whether the orbit comes back close to where it ends as a cycle or torus would do
    """
    earlier = projected[:int(len(projected) * (1.0 - exclude))]

    if len(earlier) == 0:
        return False

    extent = float(np.max(np.ptp(projected[len(projected) // 2:], axis=0)))

    if extent == 0.0:
        return True

    return float(np.min(np.linalg.norm(earlier - projected[-1], axis=1))) <= rtol * extent


def _at_rest(tail: np.ndarray, rtol: float = 1e-7) -> bool:
    steps = np.diff(tail, axis=0)

    # Step to step changes of ~multiple of 2 pi are treated as no change, e.g. hanging pendulum at +-pi flips sign when wrapping
    steps = np.where(np.abs(steps) > np.pi, steps - 2.0 * np.pi * np.round(steps / (2.0 * np.pi)), steps)

    return float(np.max(np.abs(steps))) <= rtol * max(float(np.max(np.abs(tail))), 1e-12)


def classify_dynamics(env,
                      params,
                      state,
                      *,
                      action: chex.Array | None = None,
                      policy: Callable | None = None,
                      n_steps: int = 10_000,
                      n_transient: int = 1_000,
                      n_exponents: int | None = None,
                      n_stderr: float = 3.0,
                      rtol: float = 0.01,
                      key: chex.PRNGKey | None = None,
                      ) -> DynamicsClassification:
    """
    Classifies the dynamics reached from `state` under a fixed `action` (zero control by default) or a `policy`, from
    the Lyapunov spectrum and whether the orbit stays bounded. An exponent counts as zero when it is within n_stderr
    standard errors, or rtol of the largest exponent's size, of zero. Verdicts:

    - "divergent": the orbit blows up or keeps growing, so positive exponents here are not chaos
    - "fixed point": every exponent negative (a flow, or a map whose orbit settles onto one point)
    - "periodic": a map settling onto an attracting cycle (see `period`), or a flow with one zero exponent (limit cycle)
    - "quasi-periodic": a flow with two or more zero exponents and none positive
    - "drifting": zero exponents but the orbit never recurs, e.g. a slow algebraic approach to a fixed point
    - "regular": a conservative system with no positive exponent, periodic or quasi-periodic (indistinguishable here)
    - "chaotic": one positive exponent on a bounded orbit
    - "hyperchaotic": more than one positive exponent on a bounded orbit

    `conservative` flags exponents summing to zero, e.g. a frictionless pendulum or an area preserving flow, which have
    no attractor. This is a statement about this orbit, other parameters, controls or starting states can differ.
    """
    spectrum = lyapunov_spectrum(env, params, state, action=action, policy=policy, n_steps=n_steps,
                                 n_transient=n_transient, n_exponents=n_exponents, key=key)
    dim = _StateVector(state).dim
    full_spectrum = spectrum.per_step.shape[0] == dim
    is_map = spectrum.step_duration is None

    n_tail = min(64, n_steps // 4)
    first_half, second_half, projected, tail = _orbit_summary(unwrap(env), params, state,
                                                              _action_fn(env, params, action, policy), n_transient,
                                                              n_steps, n_tail, jrandom.key(0) if key is None else key)
    first_half, second_half = float(first_half), float(second_half)
    projected, tail = np.asarray(projected), np.asarray(tail)
    bounded = bool(spectrum.finite and np.isfinite(second_half) and second_half <= 2.0 * first_half + 1e-12)

    exponents = np.asarray(spectrum.per_step)
    tolerance = np.maximum(n_stderr * np.asarray(spectrum.stderr), rtol * np.max(np.abs(exponents)))
    n_positive = int(np.sum(exponents > tolerance))
    n_zero = int(np.sum(np.abs(exponents) <= tolerance))
    conservative = bool(full_spectrum and abs(exponents.sum()) <= np.sqrt(np.sum(tolerance ** 2)) + 1e-12)

    period = None
    if not bounded:
        verdict = "divergent"
        evidence = (f"orbit not bounded (largest |state| {first_half:.3g} in the first half of the run, "
                    f"{second_half:.3g} in the second, finite {spectrum.finite})")

    elif n_positive >= 1:
        verdict = "chaotic" if n_positive == 1 else "hyperchaotic"
        evidence = f"{n_positive} positive exponent(s) on a bounded orbit, leading {exponents[0]:.4g} per step"

    elif _at_rest(tail):
        verdict = "fixed point"
        period = 1 if is_map else None
        evidence = "the orbit has come to rest"

        if n_zero:
            evidence += f", its {n_zero} zero exponent(s) are neutral directions (e.g. a symmetry), not a cycle"

    elif conservative:
        verdict = "regular"
        evidence = ("no positive exponent in a conservative system, a periodic or quasi-periodic orbit (zero exponents "
                    "come in pairs here, so the spectrum cannot tell the two apart)")

    elif is_map and n_zero == 0:
        scale = max(float(np.max(np.abs(tail))), 1e-12)
        period = int(len(np.unique(np.round(tail / (1e-6 * scale)), axis=0)))
        verdict = "fixed point" if period == 1 else "periodic"
        evidence = f"all exponents negative and the orbit repeats every {period} step(s)"

    elif n_zero == 0:
        verdict = "fixed point"
        evidence = "all exponents negative"

    elif not _recurrent(projected):
        verdict = "drifting"
        evidence = (f"{n_zero} zero exponent(s) and none positive, but the orbit never returns to where it has been, "
                    f"so it is drifting (e.g. a slow, non-exponential approach to a fixed point or boundary), not "
                    f"cycling")

    elif n_zero == 1:
        verdict = "periodic"
        evidence = "one zero exponent and the rest negative, and the orbit recurs, a limit cycle"

    else:
        verdict = "quasi-periodic"
        evidence = f"{n_zero} zero exponents and none positive, and the orbit recurs"

    if bounded and n_positive == 0 and exponents[0] > float(spectrum.stderr[0]):
        evidence += (f"; the leading exponent {exponents[0]:.3g} +- {float(spectrum.stderr[0]):.2g} is positive but "
                     f"not significant, so run longer (sticky orbits near regular islands converge slowly)")

    if not full_spectrum:
        evidence += f" (only the leading {exponents.shape[0]} of {dim} exponents tracked)"

    if conservative:
        evidence += ", exponents sum to ~0 so conservative"

    return DynamicsClassification(verdict=verdict,
                                  spectrum=spectrum,
                                  n_positive=n_positive,
                                  n_zero=n_zero,
                                  bounded=bounded,
                                  period=period,
                                  conservative=conservative,
                                  full_spectrum=full_spectrum,
                                  evidence=evidence,
                                  )


class PolicyScore(NamedTuple):
    mean_return: float  # undiscounted episode return
    std_return: float
    mean_length: float  # steps until done, at most max_steps_in_ep
    early_termination_rate: float  # fraction of episodes ending before the time limit (crashes, escapes)


class ReferenceScores(NamedTuple):
    zero: PolicyScore  # zero control, see zero_action
    random: PolicyScore  # uniformly random actions
    max_steps: int
    n_episodes: int


@partial(jax.jit, static_argnums=(0, 2, 3, 4))
def _episode_returns(base, params, random_policy: bool, max_steps: int, n_episodes: int, key):
    space = base.action_space(params)
    fixed = zero_action(base, params)

    def episode(key):
        reset_key, run_key = jrandom.split(key)
        _, state = base.reset_env(params, reset_key)

        def step(carry, key):
            state, alive, total, length = carry
            action_key, step_key = jrandom.split(key)
            action = space.sample(action_key) if random_policy else fixed
            _, _, new_state, reward, done, _ = base.step_env(action, state, params, step_key)
            total = total + jnp.where(alive, reward, 0.0)
            length = length + alive.astype(length.dtype)

            return (new_state, alive & ~done, total, length), None

        carry = (state, jnp.array(True), jnp.zeros(()), jnp.zeros((), dtype=jnp.int32)) # TODO should this be in int32?
        (_, _, total, length), _ = jax.lax.scan(step, carry, jrandom.split(run_key, max_steps))

        return total, length

    return jax.vmap(episode)(jrandom.split(key, n_episodes))


def reference_scores(env, params, *, n_episodes: int = 32, key: chex.PRNGKey | None = None) -> ReferenceScores:
    base = unwrap(env)
    max_steps = max_steps_in_ep(env, params)

    if max_steps is None:
        raise ValueError(f"{base.name} has no max_steps_in_ep, so episodes have no length")
    # TODO think this is okay, do we always want to have max_steps_in_ep idk?

    key = jrandom.key(0) if key is None else key

    def score(random_policy: bool) -> PolicyScore:
        totals, lengths = _episode_returns(base, params, random_policy, max_steps, n_episodes, key)
        totals, lengths = np.asarray(totals), np.asarray(lengths)

        return PolicyScore(mean_return=float(totals.mean()),
                           std_return=float(totals.std()),
                           mean_length=float(lengths.mean()),
                           early_termination_rate=float(np.mean(lengths < max_steps)),
                           )

    return ReferenceScores(zero=score(False), random=score(True), max_steps=max_steps, n_episodes=n_episodes)


class BifurcationSweep(NamedTuple):
    values: chex.Array  # (n_values,) the swept parameter
    # (n_values, n_samples, ...) the observable along the orbit after the transient, or with record="peaks" its
    # successive local maxima, (n_values, n_samples) padded with NaN
    samples: chex.Array
    lyapunov_per_step: chex.Array | None  # (n_values, n_exponents), largest first, None if not computed
    lyapunov_per_time: chex.Array | None  # per unit time, None for maps or if not computed
    finite: chex.Array  # (n_values,)


def _local_maxima(series: chex.Array) -> chex.Array:
    if series.ndim != 1:
        raise ValueError(f"record='peaks' needs a scalar observable, got shape {series.shape[1:]} per step")

    middle = series[1:-1]
    is_peak = (middle > series[:-2]) & (middle >= series[2:])
    slots = jnp.where(is_peak, jnp.cumsum(is_peak) - 1, series.shape[0])  # non-peaks go out of range and are dropped

    return jnp.full_like(series, jnp.nan).at[slots].set(middle, mode="drop")


def _check_sweepable(params, param_name: str):
    fields = {field.name: field for field in dataclasses.fields(params)}

    if param_name not in fields:
        raise ValueError(f"{type(params).__name__} has no field {param_name!r}, fields are {sorted(fields)}")

    if not fields[param_name].metadata.get("pytree_node", True):
        raise ValueError(f"{param_name!r} is a static field of {type(params).__name__} so it cannot be swept with "
                         f"vmap, make one env per value instead")


def bifurcation_sweep(env,
                      params,
                      state,
                      param_name: str,
                      values: chex.Array,
                      *,
                      observable: Callable | None = None,
                      action: chex.Array | None = None,
                      n_transient: int = 1_000,
                      n_samples: int = 200,
                      n_lyapunov_steps: int = 5_000,
                      n_exponents: int = 1,
                      record: str = "samples",
                      key: chex.PRNGKey | None = None,
                      ) -> BifurcationSweep:
    """
    Only non-static params fields can be swept, a static one (struct.field(False, ...)) needs one env per value.
    """

    _check_sweepable(params, param_name)
    base = unwrap(env)
    converter = _StateVector(state)
    observable = converter.to_vec if observable is None else observable
    action_fn = _action_fn(env, params, action, None)
    n_exponents = min(n_exponents, converter.dim)
    key = jrandom.key(0) if key is None else key
    burn_key, sample_key, lyapunov_key = jrandom.split(key, 3)

    def one_value(value):
        p = params.replace(**{param_name: value})

        def step(state, key):
            action_key, step_key = jrandom.split(key)
            new_state = base.step_env(action_fn(state, action_key), state, p, step_key)[2]

            return new_state, observable(new_state)

        settled, _ = jax.lax.scan(step, state, jrandom.split(burn_key, n_transient))
        settled, samples = jax.lax.scan(step, settled, jrandom.split(sample_key, n_samples))
        finite = jnp.all(jnp.isfinite(samples))

        if record == "peaks":
            samples = _local_maxima(samples)

        if n_lyapunov_steps == 0:
            return samples, jnp.zeros((n_exponents,)), finite

        log_growth, lyapunov_finite = _benettin(base, p, settled, action_fn, n_lyapunov_steps, 0, n_exponents, "jvp",
                                                1e-8, lyapunov_key)

        return samples, jnp.sort(jnp.mean(log_growth, axis=0))[::-1], finite & lyapunov_finite

    values = jnp.asarray(values)
    samples, per_step, finite = jax.jit(jax.vmap(one_value))(values)

    if n_lyapunov_steps == 0:
        per_step = None

    duration = step_duration(env, params)

    return BifurcationSweep(values=values,
                            samples=samples,
                            lyapunov_per_step=per_step,
                            lyapunov_per_time=None if per_step is None or duration is None else per_step / duration,
                            finite=finite,
                            )


class FTLEField(NamedTuple):
    per_step: chex.Array  # (n_points,) forward FTLE, log of the largest stretching over the horizon / horizon steps
    per_time: chex.Array | None  # per unit time, None for maps
    horizon: int


def ftle_field(env,
               params,
               like_state,
               points: chex.Array,
               horizon: int,
               *,
               action: chex.Array | None = None,
               key: chex.PRNGKey | None = None,
               ) -> FTLEField:
    """
    Each FTLE is log(largest singular value of the flow map Jacobian over `horizon` steps) / horizon, so ridges of
    the field are the repelling Lagrangian coherent structures of the flow. The full Jacobian is formed, so this is
    meant for low dimensional states such as particle positions. Only forward time is available, so attracting
    structures (backward FTLE) are not.
    """
    base = unwrap(env)
    converter = _StateVector(like_state)
    points = jnp.asarray(points)

    if points.ndim != 2 or points.shape[1] != converter.dim:
        raise ValueError(f"points must have shape (n_points, {converter.dim}) to match the state, got {points.shape}")

    action_fn = _action_fn(env, params, action, None)
    keys = jrandom.split(jrandom.key(0) if key is None else key, horizon)

    def flow_map(vec):
        def step(state, key):
            action_key, step_key = jrandom.split(key)
            return base.step_env(action_fn(state, action_key), state, params, step_key)[2], None

        final_state, _ = jax.lax.scan(step, converter.from_vec(vec, like_state), keys)

        return converter.to_vec(final_state)

    def ftle(vec):
        jacobian = jax.jacfwd(flow_map)(vec)
        return jnp.log(jnp.linalg.norm(jacobian, ord=2)) / horizon

    per_step = jax.jit(jax.vmap(ftle))(points.astype(converter.to_vec(like_state).dtype))
    duration = step_duration(env, params)

    return FTLEField(per_step=per_step,
                     per_time=None if duration is None else per_step / duration,
                     horizon=horizon,
                     )


@dataclasses.dataclass
class EnvSpec:
    env_id: str
    role: str
    time_structure: str
    energy: str
    forcing: str
    hyperbolicity: str
    chaotic_expected: bool | None
    note: str
    obs_dim: int
    action_dim: int
    state_dim: int
    step_duration: float | None
    max_steps: int | None
    verdict: str
    verdicts: list[str]  # one per reset
    chaotic_fraction: float | None
    evidence: str
    exponents_per_step: list[float]
    exponents_stderr: list[float]
    full_spectrum: bool
    n_positive: int
    conservative_measured: bool
    lambda1_per_step: float | None
    lambda1_per_time: float | None
    lyapunov_time_steps: float | None  # None rather than infinity when lambda_1 is not positive
    lyapunov_time: float | None
    episode_efoldings: float | None
    kaplan_yorke: float | None  # only from a full spectrum
    zero_return: float
    zero_early_termination: float
    random_return: float
    random_early_termination: float
    agrees_with_label: bool | None
    settings: dict

    @classmethod
    def from_dict(cls, data: dict) -> "EnvSpec":
        return cls(**data)


def env_spec(env_id: str,
             *,
             make_kwargs: dict | None = None,
             params_changes: dict | None = None,
             n_steps: int = 20_000,
             n_transient: int = 1_000,
             n_exponents: int | None = None,
             n_episodes: int = 32,
             n_resets: int = 4,
             reset_seed: int = 1,
             ) -> EnvSpec:
    from bifurcagym.registration import make

    labels = metadata.ENV_METADATA[env_id]
    make_kwargs = make_kwargs or {}
    params_changes = params_changes or {}
    env, params = make(env_id, cont_state=True, cont_action=True, **make_kwargs)
    params = params.replace(**params_changes)
    base = unwrap(env)
    states = [base.reset_env(params, jrandom.key(reset_seed + i))[1] for i in range(n_resets)]
    scores = reference_scores(env, params, n_episodes=n_episodes)
    expected = labels.chaotic_uncontrolled
    settings = {"make_kwargs": make_kwargs, "params_changes": params_changes, "n_steps": n_steps,
                "n_transient": n_transient, "n_exponents": n_exponents, "n_episodes": n_episodes,
                "n_resets": n_resets, "reset_seed": reset_seed}
    sizes = {"obs_dim": int(np.prod(base.observation_space(params).shape)),
             "action_dim": int(np.prod(base.action_space(params).shape)),
             "state_dim": _StateVector(states[0]).dim,
             "step_duration": step_duration(env, params),
             "max_steps": max_steps_in_ep(env, params)}
    label_fields = {"env_id": env_id, "role": labels.role, "time_structure": labels.time_structure,
                    "energy": labels.energy, "forcing": labels.forcing, "hyperbolicity": labels.hyperbolicity,
                    "chaotic_expected": expected, "note": labels.note}
    score_fields = {"zero_return": scores.zero.mean_return,
                    "zero_early_termination": scores.zero.early_termination_rate,
                    "random_return": scores.random.mean_return,
                    "random_early_termination": scores.random.early_termination_rate}

    if labels.energy == "stochastic":
        # Exponents of a noise driven system measure something else (a random dynamical system), so no verdict
        return EnvSpec(**label_fields, **sizes, verdict="stochastic (not assessed)", verdicts=[],
                       chaotic_fraction=None, evidence="noise driven, a Lyapunov verdict does not apply",
                       exponents_per_step=[], exponents_stderr=[], full_spectrum=False, n_positive=0,
                       conservative_measured=False, lambda1_per_step=None, lambda1_per_time=None,
                       lyapunov_time_steps=None, lyapunov_time=None, episode_efoldings=None, kaplan_yorke=None,
                       **score_fields, agrees_with_label=None, settings=settings)

    results = [classify_dynamics(env, params, state, n_steps=n_steps, n_transient=n_transient,
                                 n_exponents=n_exponents) for state in states]
    verdicts = [result.verdict for result in results]
    chaotic = [verdict in ("chaotic", "hyperchaotic") for verdict in verdicts]
    chaotic_fraction = float(np.mean(chaotic))

    if len(set(verdicts)) == 1:
        verdict = verdicts[0]

    elif any(chaotic):
        verdict = f"mixed ({sum(chaotic)}/{len(chaotic)} chaotic)"

    else:
        verdict = max(set(verdicts), key=verdicts.count)

    order = np.argsort([float(result.spectrum.per_step[0]) for result in results])
    classification = results[int(order[len(order) // 2])]
    spectrum = classification.spectrum
    lyapunov_steps, lyapunov_seconds = lyapunov_time(spectrum)
    measured_chaotic = chaotic_fraction > 0

    return EnvSpec(**label_fields,
                   **sizes,
                   verdict=verdict,
                   verdicts=verdicts,
                   chaotic_fraction=chaotic_fraction,
                   evidence=classification.evidence,
                   exponents_per_step=[float(v) for v in spectrum.per_step],
                   exponents_stderr=[float(v) for v in spectrum.stderr],
                   full_spectrum=classification.full_spectrum,
                   n_positive=classification.n_positive,
                   conservative_measured=classification.conservative,
                   lambda1_per_step=float(spectrum.per_step[0]),
                   lambda1_per_time=None if spectrum.per_time is None else float(spectrum.per_time[0]),
                   lyapunov_time_steps=None if not np.isfinite(lyapunov_steps) else float(lyapunov_steps),
                   lyapunov_time=None if lyapunov_seconds is None or not np.isfinite(lyapunov_seconds)
                   else float(lyapunov_seconds),
                   episode_efoldings=episode_efoldings(env, params, spectrum),
                   kaplan_yorke=kaplan_yorke_dimension(spectrum.per_step) if classification.full_spectrum else None,
                   **score_fields,
                   agrees_with_label=None if expected is None else bool(measured_chaotic == expected),
                   settings=settings,
                   )


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


_TABLE_COLUMNS = (("Env", lambda s: s.env_id),
                  ("Role", lambda s: s.role),
                  ("Structure", lambda s: s.time_structure),
                  ("Energy", lambda s: s.energy),
                  ("Forcing", lambda s: s.forcing),
                  ("Hyperbolicity", lambda s: s.hyperbolicity),
                  ("Measured", lambda s: s.verdict + ("" if s.agrees_with_label is not False else " (!)")),
                  ("λ₁ / step", lambda s: _fmt(s.lambda1_per_step)),
                  ("λ₁ / time", lambda s: _fmt(s.lambda1_per_time)),
                  ("Lyapunov time", lambda s: _fmt(s.lyapunov_time if s.lyapunov_time is not None
                                                   else s.lyapunov_time_steps)),
                  ("e-folds / episode", lambda s: _fmt(s.episode_efoldings)),
                  ("KY dim", lambda s: _fmt(s.kaplan_yorke)),
                  ("Zero ctrl return", lambda s: _fmt(s.zero_return)),
                  ("Random return", lambda s: _fmt(s.random_return)),
                  )


def spec_table(specs: list[EnvSpec], fmt: str = "markdown") -> str:
    headers = [name for name, _ in _TABLE_COLUMNS]
    rows = [[str(getter(spec)) for _, getter in _TABLE_COLUMNS] for spec in specs]

    if fmt == "markdown":
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join([":--"] * 6 + [":--:"] + ["--:"] * (len(headers) - 7)) + "|"]
        lines += ["| " + " | ".join([f"`{row[0]}`"] + row[1:]) + " |" for row in rows]

        return "\n".join(lines)

    if fmt == "latex":
        def escape(text: str) -> str:
            return (text.replace("_", r"\_").replace("&", r"\&").replace("λ₁", r"$\lambda_1$")
                    .replace("–", "--"))
        lines = [r"\begin{tabular}{" + "l" * 7 + "r" * (len(headers) - 7) + "}", r"\toprule",
                 " & ".join(escape(h) for h in headers) + r" \\", r"\midrule"]
        lines += [" & ".join(escape(cell) for cell in row) + r" \\" for row in rows]
        lines += [r"\bottomrule", r"\end{tabular}"]

        return "\n".join(lines)

    raise ValueError(f"fmt must be 'markdown' or 'latex', got {fmt!r}")
