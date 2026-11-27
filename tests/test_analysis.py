"""
Tests for bifurcagym.analysis against published Lyapunov exponents and exact identities.
"""

import dataclasses
import json

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

import bifurcagym
from bifurcagym import analysis


def spectrum_for(env_id, n_steps, params_changes=None, make_kwargs=None, **kwargs):
    env, params = bifurcagym.make(env_id, cont_state=True, cont_action=True, **(make_kwargs or {}))
    params = params.replace(**(params_changes or {}))
    _, state = env.reset(params, jrandom.key(1))
    return env, params, analysis.lyapunov_spectrum(env, params, state, n_steps=n_steps, **kwargs)


# # # Spectra against known values
########################################################################################################################
def test_henon_spectrum_and_kaplan_yorke_dimension():
    """Sprott (2003): (0.41922, -1.62319), Kaplan-Yorke dimension 1.258"""
    _, _, spectrum = spectrum_for("HenonMap-v0", 50_000)

    assert spectrum.finite
    np.testing.assert_allclose(spectrum.per_step, (0.41922, -1.62319), atol=0.01)
    assert spectrum.per_time is None  # a map has no time unit
    np.testing.assert_allclose(analysis.kaplan_yorke_dimension(spectrum.per_step), 1.258, atol=0.01)
    assert analysis.n_positive_exponents(spectrum) == 1


@pytest.mark.parametrize("env_id, make_kwargs, params_changes, expected",
                         [("LogisticMap-v0", {"init_r": 4.0}, {}, np.log(2.0)),  # exact at r = 4
                          ("TentMap-v0", {}, {"init_mu": 1.9}, np.log(1.9)),  # |f'| = mu everywhere
                          ])
def test_one_dimensional_maps(env_id, make_kwargs, params_changes, expected):
    _, _, spectrum = spectrum_for(env_id, 50_000, params_changes=params_changes, make_kwargs=make_kwargs)
    np.testing.assert_allclose(spectrum.per_step, (expected,), atol=0.01)


def test_lorenz_spectrum_per_unit_time():
    """Sprott (2003): (0.9056, 0, -14.5723) per unit time, Kaplan-Yorke dimension 2.062"""
    env, params, spectrum = spectrum_for("Lorenz63-v0", 50_000)

    assert spectrum.step_duration == env.dt
    np.testing.assert_allclose(spectrum.per_time, (0.9056, 0.0, -14.5723), atol=0.03)
    np.testing.assert_allclose(analysis.kaplan_yorke_dimension(spectrum.per_step), 2.062, atol=0.01)


def test_exact_tangents_agree_with_finite_differences():
    _, _, exact = spectrum_for("Lorenz63-v0", 20_000)
    _, _, finite_difference = spectrum_for("Lorenz63-v0", 20_000, method="fd")

    np.testing.assert_allclose(exact.per_step, finite_difference.per_step, atol=1e-4)


def test_leading_exponents_only():
    _, _, spectrum = spectrum_for("Lorenz63-v0", 20_000, n_exponents=1)

    assert spectrum.per_step.shape == (1,)
    np.testing.assert_allclose(spectrum.per_time, (0.9056,), atol=0.03)


def test_closed_loop_with_a_zero_policy_matches_open_loop():
    env, params = bifurcagym.make("HenonMap-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(1))
    open_loop = analysis.lyapunov_spectrum(env, params, state, n_steps=5_000)
    closed_loop = analysis.lyapunov_spectrum(env, params, state, n_steps=5_000,
                                             policy=lambda obs, key: jnp.zeros(2) * obs[0])

    np.testing.assert_allclose(open_loop.per_step, closed_loop.per_step, rtol=1e-12)


def test_large_states_need_n_exponents(monkeypatch):
    monkeypatch.setattr(analysis, "_MAX_DEFAULT_EXPONENTS", 2)
    env, params = bifurcagym.make("Lorenz63-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(1))

    with pytest.raises(ValueError, match="n_exponents"):
        analysis.lyapunov_spectrum(env, params, state, n_steps=10)


# # # Env helpers
########################################################################################################################
def test_unwrap_reaches_the_base_env_through_every_wrapper():
    env, params = bifurcagym.make("Pendulum-v0", cont_state=True, cont_action=True, normalised=True, autoreset=True,
                                  metrics=True)
    base = analysis.unwrap(env)

    assert type(base).__name__ == "PendulumCSCA"
    assert analysis.unwrap(base) is base
    assert analysis.step_duration(env, params) == base.dt
    assert analysis.max_steps_in_ep(env, params) == base.max_steps_in_ep


def test_zero_action():
    env, params = bifurcagym.make("HenonMap-v0", cont_state=True, cont_action=True)
    discrete_env, discrete_params = bifurcagym.make("HenonMap-v0", cont_state=True, cont_action=False)

    np.testing.assert_array_equal(analysis.zero_action(env, params), jnp.zeros(2))
    np.testing.assert_array_equal(analysis.zero_action(discrete_env, discrete_params), jnp.zeros(1, dtype=int))


def test_state_vector_round_trip_with_complex_leaves():
    """HydroGym's spectral states are complex, they become real and imaginary parts and the time counter is kept"""
    state = {"time": jnp.array(3), "u": jnp.arange(3.0), "omega": jnp.array((1 + 2j, 3 - 4j))}
    converter = analysis._StateVector(state)

    vec = converter.to_vec(state)
    assert converter.dim == 7 and vec.shape == (7,) and not jnp.iscomplexobj(vec)
    rebuilt = converter.from_vec(vec, state)
    np.testing.assert_array_equal(rebuilt["u"], state["u"])
    np.testing.assert_array_equal(rebuilt["omega"], state["omega"])
    assert rebuilt["time"] is state["time"]


# # # Quantities derived from a spectrum
########################################################################################################################
@pytest.mark.parametrize("exponents, expected",
                         [((1.0, -2.0), 1.5),
                          ((-0.1, -1.0), 0.0),  # contracting, a stable fixed point
                          ((0.5, 0.0, -0.5), 3.0),  # conservative, the full dimension
                          ((0.9056, 0.0, -14.5723), 2.0 + 0.9056 / 14.5723),
                          ])
def test_kaplan_yorke_dimension(exponents, expected):
    np.testing.assert_allclose(analysis.kaplan_yorke_dimension(jnp.array(exponents)), expected)


def test_lyapunov_time_and_episode_efoldings():
    env, params, spectrum = spectrum_for("Lorenz63-v0", 20_000)
    steps, seconds = analysis.lyapunov_time(spectrum)

    np.testing.assert_allclose(steps, 1.0 / float(spectrum.per_step[0]))
    np.testing.assert_allclose(seconds, steps * env.dt)
    np.testing.assert_allclose(analysis.episode_efoldings(env, params, spectrum),
                               env.max_steps_in_ep * float(spectrum.per_step[0]))


# # # Classifying the dynamics
########################################################################################################################
@pytest.mark.parametrize("r, verdict, period", [(2.8, "fixed point", 1), (3.2, "periodic", 2), (3.5, "periodic", 4),
                                                (3.83, "periodic", 3), (4.0, "chaotic", None)])
def test_classify_logistic_regimes(r, verdict, period):
    env, params = bifurcagym.make("LogisticMap-v0", cont_state=True, cont_action=True)
    params = params.replace(init_r=r)
    _, state = env.reset(params, jrandom.key(1))

    result = analysis.classify_dynamics(env, params, state)
    assert (result.verdict, result.period) == (verdict, period)


@pytest.mark.parametrize("env_id, params_changes, verdict, conservative",
                         [("HenonMap-v0", {}, "chaotic", False),
                          ("HenonMap-v0", {"init_a": 2.0}, "divergent", False),  # escapes to infinity, not chaos
                          ("Lorenz63-v0", {}, "chaotic", False),
                          ("Lorenz63-v0", {"rho": 10.0}, "fixed point", False),
                          ("Pendulum-v0", {"max_speed": 1e6}, "regular", True),  # frictionless swing
                          ("DoubleGyreFlow-v0", {}, "chaotic", True),  # area preserving flow
                          ])
def test_classify_dynamics(env_id, params_changes, verdict, conservative):
    env, params = bifurcagym.make(env_id, cont_state=True, cont_action=True)
    params = params.replace(**params_changes)
    _, state = env.reset(params, jrandom.key(1))

    result = analysis.classify_dynamics(env, params, state, n_steps=20_000)
    assert result.verdict == verdict, result.evidence
    assert result.conservative == conservative
    assert result.full_spectrum


def test_classify_reports_a_truncated_spectrum():
    env, params = bifurcagym.make("Lorenz63-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(1))

    result = analysis.classify_dynamics(env, params, state, n_exponents=1)
    assert result.verdict == "chaotic" and not result.full_spectrum and not result.conservative
    assert "only the leading 1 of 3" in result.evidence


# # # Bifurcation sweeps
########################################################################################################################
def test_logistic_period_doubling_and_chaos():
    """Fixed point, period 2, period 4, the period 3 window, then chaos, with lambda = ln|2 - r| on the fixed point"""
    env, params = bifurcagym.make("LogisticMap-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(1))
    sweep = analysis.bifurcation_sweep(env, params, state, "init_r", jnp.array((2.8, 3.2, 3.5, 3.83, 3.9, 4.0)),
                                       n_transient=5_000, n_samples=64, n_lyapunov_steps=20_000)

    distinct = [len(np.unique(np.round(np.asarray(samples), 6))) for samples in sweep.samples]
    leading = np.asarray(sweep.lyapunov_per_step[:, 0])
    assert bool(jnp.all(sweep.finite))
    assert distinct[:4] == [1, 2, 4, 3]
    assert np.all(leading[:4] < 0) and np.all(leading[4:] > 0)
    np.testing.assert_allclose(leading[0], np.log(0.8), atol=1e-6)
    np.testing.assert_allclose(leading[-1], np.log(2.0), atol=0.01)


def test_lorenz_becomes_chaotic_with_rho():
    """The fixed points C+- are stable below rho ~ 24.74, chaos at the classic rho = 28"""
    env, params = bifurcagym.make("Lorenz63-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(1))
    sweep = analysis.bifurcation_sweep(env, params, state, "rho", jnp.array((10.0, 20.0, 28.0)), n_samples=10,
                                       n_lyapunov_steps=20_000)

    leading = np.asarray(sweep.lyapunov_per_time[:, 0])
    assert leading[0] < 0 and leading[1] < 0
    np.testing.assert_allclose(leading[2], 0.9056, atol=0.05)


@pytest.mark.parametrize("param_name, message", [("maximum_max_control", "static field"), ("nope", "no field")])
def test_sweep_rejects_static_and_unknown_fields(param_name, message):
    env, params = bifurcagym.make("LogisticMap-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(1))

    with pytest.raises(ValueError, match=message):
        analysis.bifurcation_sweep(env, params, state, param_name, jnp.array((1.0,)))


# # # Finite time Lyapunov exponents
########################################################################################################################
def test_tent_map_ftle_is_exactly_ln_mu():
    """|f'| = mu everywhere, so every finite time exponent is ln mu exactly"""
    env, params = bifurcagym.make("TentMap-v0", cont_state=True, cont_action=True)
    params = params.replace(init_mu=1.9)
    _, state = env.reset(params, jrandom.key(1))
    points = jrandom.uniform(jrandom.key(2), (100, 1), minval=0.05, maxval=0.95)

    field = analysis.ftle_field(env, params, state, points, horizon=10)
    np.testing.assert_allclose(field.per_step, np.log(1.9), rtol=1e-12)
    assert field.per_time is None


def test_double_gyre_ftle_has_a_ridge():
    """Shadden, Lekien and Marsden (2005): values up to ~0.5 on a thin repelling ridge, small inside the gyres"""
    env, params = bifurcagym.make("DoubleGyreFlow-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(1))
    xs, ys = jnp.meshgrid(jnp.linspace(0.01, 1.99, 61), jnp.linspace(0.01, 0.99, 31))
    horizon = int(round(15 / analysis.unwrap(env).dt))

    field = analysis.ftle_field(env, params, state.replace(time=0), jnp.stack((xs.ravel(), ys.ravel()), axis=1),
                                horizon)
    ftle = np.asarray(field.per_time)
    assert np.all(np.isfinite(ftle))
    assert 0.3 < ftle.max() < 1.0
    assert ftle.max() > 2.5 * np.median(ftle)


def test_ftle_points_must_match_the_state():
    env, params = bifurcagym.make("DoubleGyreFlow-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(1))

    with pytest.raises(ValueError, match="points must have shape"):
        analysis.ftle_field(env, params, state, jnp.zeros((5, 3)), horizon=2)


def test_lyapunov_time_is_infinite_without_chaos():
    contracting = analysis.LyapunovSpectrum(per_step=jnp.array((-0.1,)), per_time=None, stderr=jnp.zeros(1),
                                            step_duration=None, n_steps=1, finite=True)

    assert analysis.lyapunov_time(contracting) == (np.inf, None)


# # # Classifier details
########################################################################################################################
def test_at_rest_ignores_an_angle_flipping_sign_at_pi():
    """A pendulum resting at +-pi wraps between +pi and -pi each step, which is not motion"""
    resting = np.tile(np.array((0.3, np.pi, 0.0)), (64, 1))
    resting[1::2, 1] = -np.pi
    moving = resting.copy()
    moving[:, 0] = np.linspace(0.0, 1.0, 64)

    assert analysis._at_rest(resting)
    assert not analysis._at_rest(moving)


def test_recurrence_tells_a_cycle_from_a_drift():
    t = np.linspace(0.0, 40.0 * np.pi, 4_000)
    cycle = np.stack((np.cos(t), np.sin(t)), axis=1)
    drift = np.stack((1.0 / (1.0 + t), 0.5 / (1.0 + t)), axis=1)  # algebraic approach to the origin

    assert analysis._recurrent(cycle)
    assert not analysis._recurrent(drift)


def test_zero_action_is_clipped_into_the_action_box():
    class Box:
        low, high, shape, dtype = jnp.array((0.5, 0.0)), jnp.array((1.0, 1.0)), (2,), jnp.float64

    class Env:
        def action_space(self, params):
            return analysis.spaces.Box(Box.low, Box.high, Box.shape, Box.dtype)

    np.testing.assert_array_equal(analysis.zero_action(Env(), None), (0.5, 0.0))


# # # Reference scores, metadata and env specs
########################################################################################################################
def test_reference_scores():
    """CartPole: zero force never leaves the track, random forcing always does"""
    env, params = bifurcagym.make("CartPole-v0", cont_state=True, cont_action=True)
    scores = analysis.reference_scores(env, params, n_episodes=8)

    assert scores.max_steps == analysis.max_steps_in_ep(env, params) and scores.n_episodes == 8
    assert scores.zero.early_termination_rate == 0.0 and scores.zero.mean_length == scores.max_steps
    assert scores.random.early_termination_rate == 1.0 and scores.random.mean_length < scores.max_steps
    assert 0.0 <= scores.zero.mean_return <= scores.max_steps  # rewards are in [0, 1] per step


def test_every_registered_env_has_valid_metadata():
    analysis.metadata.validate()
    assert set(analysis.metadata.ENV_METADATA) == set(bifurcagym.registered_envs)


def test_env_spec_round_trips_through_json_and_renders():
    spec = analysis.env_spec("TentMap-v0", n_steps=2_000, n_episodes=2, n_resets=2)

    assert spec.verdict == "chaotic" and spec.verdicts == ["chaotic", "chaotic"] and spec.agrees_with_label
    np.testing.assert_allclose(spec.lambda1_per_step, np.log(1.999), atol=0.02)
    assert analysis.EnvSpec.from_dict(json.loads(json.dumps(dataclasses.asdict(spec)))) == spec

    markdown = analysis.spec_table([spec], fmt="markdown")
    latex = analysis.spec_table([spec], fmt="latex")
    assert "`TentMap-v0`" in markdown and markdown.count("\n") == 2
    assert r"\begin{tabular}" in latex and "TentMap-v0" in latex
    with pytest.raises(ValueError, match="fmt"):
        analysis.spec_table([spec], fmt="html")


def test_env_spec_does_not_assess_noise_driven_envs():
    spec = analysis.env_spec("WetChicken-v0", n_steps=100, n_episodes=2)

    assert spec.verdict == "stochastic (not assessed)" and spec.lambda1_per_step is None
    assert spec.agrees_with_label is None and spec.exponents_per_step == []
