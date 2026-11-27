"""
Tests for record="peaks" in bifurcation_sweep, the bifurcation diagram defaults and bifurcagym.plotting.
"""

import dataclasses

import jax.numpy as jnp
import jax.random as jrandom
import matplotlib
import numpy as np
import pytest

import bifurcagym
from bifurcagym import analysis
from bifurcagym import metadata
from bifurcagym import plotting


matplotlib.use("Agg")


# # # Peaks
########################################################################################################################
def test_local_maxima_in_order_and_nan_padded():
    series = jnp.array((0.0, 2.0, 1.0, 3.0, 3.0, 0.5, 4.0, 1.0))
    peaks = analysis._local_maxima(series)

    np.testing.assert_array_equal(peaks[:3], (2.0, 3.0, 4.0))  # a plateau counts once, at its first point
    assert peaks.shape == series.shape and np.all(np.isnan(np.asarray(peaks[3:])))


def test_peaks_need_a_scalar_observable():
    env, params = bifurcagym.make("Lorenz63-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(0))

    with pytest.raises(ValueError, match="scalar observable"):
        analysis.bifurcation_sweep(env, params, state, "rho", jnp.array((28.0,)), record="peaks", n_lyapunov_steps=0)
    with pytest.raises(ValueError, match="record"):
        analysis.bifurcation_sweep(env, params, state, "rho", jnp.array((28.0,)), record="minima")


def test_lorenz_map_is_single_humped():
    """Successive maxima of z at rho = 28 lie on Lorenz's (1963) cusp shaped map, rising then falling"""
    env, params = bifurcagym.make("Lorenz63-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(0))
    sweep = analysis.bifurcation_sweep(env, params, state, "rho", jnp.array((28.0,)), observable=lambda s: s.x[2],
                                       record="peaks", n_transient=2_000, n_samples=50_000, n_lyapunov_steps=0)

    peaks = np.asarray(sweep.samples[0])
    peaks = peaks[np.isfinite(peaks)]
    current, following = peaks[:-1], peaks[1:]
    cusp = current[np.argmax(following)]
    assert len(peaks) > 500
    assert 0.3 < (cusp - current.min()) / np.ptp(current) < 0.7
    assert np.corrcoef(current[current < cusp], following[current < cusp])[0, 1] > 0.95
    assert np.corrcoef(current[current > cusp], following[current > cusp])[0, 1] < -0.95


# # # Defaults and plots
########################################################################################################################
@pytest.mark.parametrize("env_id", metadata.BIFURCATION_DEFAULTS.keys())
def test_bifurcation_defaults_name_a_sweepable_param(env_id):
    defaults = metadata.BIFURCATION_DEFAULTS[env_id]
    _, params = bifurcagym.make(env_id, cont_state=True, cont_action=True)

    assert env_id in bifurcagym.registered_envs and defaults.record in ("samples", "peaks")
    assert defaults.low < defaults.high
    analysis._check_sweepable(params, defaults.param)
    assert defaults.param in {field.name for field in dataclasses.fields(params)}


def test_bifurcation_diagramme_renders_the_logistic_cascade():
    ax, twin, sweep = plotting.bifurcation_diagramme("LogisticMap-v0", n_values=30)

    assert sweep.samples.shape == (30, metadata.BIFURCATION_DEFAULTS["LogisticMap-v0"].n_samples)
    assert len(ax.collections) == 1 and twin is not None and ax.get_title() == "LogisticMap-v0"
    leading = np.asarray(sweep.lyapunov_per_step[:, 0])
    assert leading[0] < 0 < leading[-1]  # periodic at r = 2.8, chaotic at r = 4


def test_bifurcation_diagramme_rejects_an_env_without_defaults():
    with pytest.raises(ValueError, match="no bifurcation defaults"):
        plotting.bifurcation_diagramme("CartPole-v0")


def test_ftle_and_return_map_render():
    env, params = bifurcagym.make("DoubleGyreFlow-v0", cont_state=True, cont_action=True)
    _, state = env.reset(params, jrandom.key(0))
    xs, ys = np.meshgrid(np.linspace(0.05, 1.95, 8), np.linspace(0.05, 0.95, 4))
    field = analysis.ftle_field(env, params, state, jnp.stack((xs.ravel(), ys.ravel()), axis=1), horizon=20)

    assert plotting.plot_ftle(field, xs, ys).collections
    peaks = np.array(((1.0, 2.0, 3.0, np.nan), (2.0, 1.0, np.nan, np.nan)))
    ax = plotting.plot_return_map(peaks)
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), ((1.0, 2.0), (2.0, 3.0), (2.0, 1.0)))
