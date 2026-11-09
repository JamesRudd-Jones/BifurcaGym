import gc
import itertools
import os

# Must be set before jax is imported, by default every xdist worker would start thread pools sized to the whole machine
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NPROC", "2")  # size of the XLA CPU client thread pools
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(thread_var, "1")

import jax  # noqa: E402
import pytest  # noqa: E402

# Must be set before bifurcagym is imported, as env param defaults are built as jnp arrays at import time
jax.config.update("jax_enable_x64", True)

import bifurcagym  # noqa: E402

# Clearing costs roughly 0.3 s as JAX's internal functions recompile, so it is not done after every test
CLEAR_JAX_CACHES_EVERY = 10
_test_counter = itertools.count(1)


@pytest.fixture(autouse=True)
def release_compiled_envs():
    """The env methods jit with self as a static argument, so JAX keeps the compiled code of every env instance made"""
    yield
    if next(_test_counter) % CLEAR_JAX_CACHES_EVERY == 0:
        jax.clear_caches()
        gc.collect()


# Envs that take minutes per test on CPU, run with `pytest -m slow` or `pytest -m ""`
SLOW_ENVS = ["FluidicPinball-v0",
             "HydroGym-Channel-v0",
             "Hopper-v0",  # ~40 s jit compile per env instance
             "Walker2d-v0",
             ]

# Envs whose observations are a partial view of the state, so generative_step cannot rebuild the state
PARTIALLY_OBSERVABLE_ENVS = ["KS-v0",
                             "FluidicPinball-v0",
                             "HydroGym-Kolmogorov-v0",
                             "HydroGym-Channel-v0",
                             "Hopper-v0",  # forward position not observed and velocities clipped, as in Gymnasium
                             "Walker2d-v0",
                             ]

# Cheaper solver settings so the HydroGym envs run the full suite in reasonable time, physics is not checked here
TEST_ENV_KWARGS = {"HydroGym-Kolmogorov-v0": {"dt": 1e-2, "action_time": 1.0},
                   "HydroGym-Channel-v0": {"dns_steps_per_env_step": 1},
                   }


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
    """Skips only for state/action combinations that an env deliberately does not support, or missing optional deps"""
    kwargs = {**TEST_ENV_KWARGS.get(env_name, {}), **kwargs}
    try:
        return bifurcagym.make(env_name, cont_state=cont_state, cont_action=cont_action, **kwargs)
    except ValueError as e:
        if str(e).startswith(("No ", "Can't Normalise")):
            pytest.skip(f"Unsupported combination: {e}")
        raise
    except ImportError as e:
        if "hydrogym" in str(e) or "mujoco" in str(e):
            pytest.skip(f"Optional dependency missing: {e}")
        raise


# Discrete state envs whose obs exactly encode the state (others project a continuous state onto a grid)
EXACT_DISCRETE_STATE_ENVS = ["WetChicken-v0",
                             ]


def supports_generative_step(env_name, cont_state):
    if not cont_state:
        return env_name in EXACT_DISCRETE_STATE_ENVS
    return env_name not in PARTIALLY_OBSERVABLE_ENVS
