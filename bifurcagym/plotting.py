from collections.abc import Sequence

import jax.random as jrandom
import numpy as np

from bifurcagym import analysis
from bifurcagym import metadata

import cmocean


def plot_bifurcation_diagramme(sweep: analysis.BifurcationSweep, param_name: str = "parameter",
                               observable_name: str = "observable", ax=None, lyapunov: bool = True,
                               point_size: float = 0.3):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(8, 5))

    values = np.asarray(sweep.values)
    samples = np.asarray(sweep.samples).reshape(values.shape[0], -1)
    xs = np.repeat(values, samples.shape[1])
    ys = samples.ravel()
    keep = np.isfinite(ys)

    ax.scatter(xs[keep], ys[keep], s=point_size, c="black", marker=".", linewidths=0, rasterized=True)
    ax.set_xlabel(param_name)
    ax.set_ylabel(observable_name)
    ax.set_xlim(values.min(), values.max())

    if not lyapunov or sweep.lyapunov_per_step is None:
        return ax, None

    per_time = sweep.lyapunov_per_time is not None
    leading = np.asarray(sweep.lyapunov_per_time if per_time else sweep.lyapunov_per_step)[:, 0]
    twin = ax.twinx()
    twin.plot(values, leading, color="tab:red", linewidth=0.8)
    twin.axhline(0.0, color="tab:red", linewidth=0.5, linestyle="--")
    twin.set_ylabel("λ₁ per unit time" if per_time else "λ₁ per step", color="tab:red")
    twin.tick_params(axis="y", colors="tab:red")

    return ax, twin


def plot_ftle(field: analysis.FTLEField, xs: np.ndarray, ys: np.ndarray, ax=None, cmap: str = cmocean.cm.thermal):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(8, 5))

    values = np.asarray(field.per_time if field.per_time is not None else field.per_step).reshape(np.shape(xs))
    mesh = ax.pcolormesh(xs, ys, values, cmap=cmap, shading="auto")
    ax.figure.colorbar(mesh, ax=ax, label="FTLE per unit time" if field.per_time is not None else "FTLE per step")
    ax.set_aspect("equal")

    return ax


def plot_return_map(peaks: np.ndarray, ax=None, point_size: float = 1.0, observable_name: str = "peak"):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(8, 5))

    peaks = np.atleast_2d(np.asarray(peaks))
    current = np.concatenate([row[np.isfinite(row)][:-1] for row in peaks])
    following = np.concatenate([row[np.isfinite(row)][1:] for row in peaks])

    ax.scatter(current, following, s=point_size, c="black", marker=".", linewidths=0)
    ax.set_xlabel(f"{observable_name} n")
    ax.set_ylabel(f"{observable_name} n + 1")

    return ax


def bifurcation_diagramme(env_id: str, values: Sequence[float] | None = None, n_values: int = 400, ax=None,
                        reset_seed: int = 0, **sweep_kwargs):
    from bifurcagym.registration import make

    if env_id not in metadata.BIFURCATION_DEFAULTS:
        raise ValueError(f"no bifurcation defaults for {env_id}, has {sorted(metadata.BIFURCATION_DEFAULTS)}")

    defaults = metadata.BIFURCATION_DEFAULTS[env_id]
    env, params = make(env_id, cont_state=True, cont_action=True)
    _, state = analysis.unwrap(env).reset_env(params, jrandom.key(reset_seed))
    values = np.linspace(defaults.low, defaults.high, n_values) if values is None else np.asarray(values)
    kwargs = {"observable": defaults.observable, "record": defaults.record, "n_transient": defaults.n_transient,
              "n_samples": defaults.n_samples, "n_lyapunov_steps": defaults.n_lyapunov_steps, **sweep_kwargs}

    sweep = analysis.bifurcation_sweep(env, params, state, defaults.param, values, **kwargs)
    ax, twin = plot_bifurcation_diagramme(sweep, defaults.param, defaults.observable_name, ax=ax)
    ax.set_title(env_id)

    return ax, twin, sweep
