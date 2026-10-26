import itertools

import jax
import pytest

# Must be set before bifurcagym is imported, as env param defaults are built as jnp arrays at import time
jax.config.update("jax_enable_x64", True)

import bifurcagym  # noqa: E402


# Envs that take minutes per test on CPU, run with `pytest -m slow` or `pytest -m ""`
SLOW_ENVS = ["FluidicPinball-v0",
             ]

# Envs whose observations are a partial view of the state, so generative_step cannot rebuild the state
PARTIALLY_OBSERVABLE_ENVS = ["KS-v0",
                             "FluidicPinball-v0",
                             ]


def env_param(env_name, *args):
    marks = [pytest.mark.slow] if env_name in SLOW_ENVS else []
    return pytest.param(env_name, *args, marks=marks)


def env_combinations(*extra_axes):
    """All registered envs crossed with cont_state, cont_action and any extra boolean axes"""
    axes = [[True, False], [True, False]] + [[True, False] for _ in extra_axes]
    return [env_param(env_name, *combo)
            for env_name in bifurcagym.registered_envs
            for combo in itertools.product(*axes)]


def make_or_skip(env_name, cont_state, cont_action, **kwargs):
    """Skips only for state/action combinations that an env deliberately does not support"""
    try:
        return bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action, **kwargs)
    except ValueError as e:
        if str(e).startswith(("No ", "Can't Normalise")):
            pytest.skip(f"Unsupported combination: {e}")
        raise


# Discrete state envs whose obs exactly encode the state (others project a continuous state onto a grid)
EXACT_DISCRETE_STATE_ENVS = ["WetChicken-v0",
                             ]


def supports_generative_step(env_name, cont_state):
    if not cont_state:
        return env_name in EXACT_DISCRETE_STATE_ENVS
    return env_name not in PARTIALLY_OBSERVABLE_ENVS
