"""
Every env step is differentiable with respect to its state. The analysis module relies on this for exact tangents
(jax.jvp) in its Lyapunov exponents and FTLE fields, and differentiable simulation methods need it too. A
jax.lax.stop_gradient on the step outputs would make every tangent exactly zero.
"""

from conftest import env_param
from conftest import make_or_skip
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jrandom
import pytest

import bifurcagym


def split_float_leaves(state):
    """
    Flattens the real and complex float leaves of a state into a vector (HydroGym's spectral states are complex), the
    integer time counter and any PRNG keys are left alone
    """
    leaves, treedef = jax.tree.flatten(state)
    is_float = [jnp.issubdtype(jnp.result_type(leaf), jnp.inexact) for leaf in leaves]
    vec, unravel = ravel_pytree([leaf for leaf, f in zip(leaves, is_float, strict=True) if f])

    def rebuild(new_vec):
        floats = iter(unravel(new_vec))
        return treedef.unflatten([next(floats) if f else leaf for leaf, f in zip(leaves, is_float, strict=True)])

    return vec, rebuild


@pytest.mark.parametrize("env_name", [env_param(env_name) for env_name in bifurcagym.registered_envs])
def test_step_is_differentiable_in_the_state(env_name):
    env, params = make_or_skip(env_name, cont_state=True, cont_action=True)
    reset_key, action_key, step_key, tangent_key = jrandom.split(jrandom.key(0), 4)
    _, state = env.reset(params, reset_key)
    action = env.action_space(params).sample(action_key)

    state_vec, rebuild = split_float_leaves(state)

    def next_state_vec(vec):
        _, _, new_state, _, _, _ = env.step(action, rebuild(vec), params, step_key)
        return split_float_leaves(new_state)[0]

    tangent = jrandom.normal(tangent_key, state_vec.shape, dtype=state_vec.dtype)
    _, out_tangent = jax.jvp(next_state_vec, (state_vec,), (tangent,))

    assert jnp.all(jnp.isfinite(out_tangent)), f"{env_name}: non-finite tangent through the step"
    assert float(jnp.linalg.norm(out_tangent)) > 0.0, f"{env_name}: zero tangent, is a stop_gradient blocking it?"
