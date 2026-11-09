"""
Physical validity of the MJX Hopper and Walker2d: agreement with CPU MuJoCo, dissipation without actuation, and chaos
under sustained actuation.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from bifurcagym.envs.locomotion import mujoco_locomotion  # noqa: E402
from physics_utils import lyapunov_spectrum  # noqa: E402


pytestmark = pytest.mark.physics

ENVS = [mujoco_locomotion.HopperCSCA, mujoco_locomotion.Walker2dCSCA]
FORCING_FREQUENCY = 1.5  # Hz, roughly a gait frequency


def rollout(env, params, state, actions):
    def body(state, action):
        _, _, new_state, _, _, _ = env.step(action, state, params, jrandom.key(0))
        return new_state, new_state
    return jax.lax.scan(body, state, actions)


@pytest.mark.parametrize("env_cls", ENVS)
def test_matches_cpu_mujoco(env_cls):
    """One step from states along a random actuation rollout, including falls and ground contact, matches mj_step"""
    env = env_cls()
    params = env.default_params.replace(terminate_when_unhealthy=False)
    _, state = env.reset(params, jrandom.key(0))
    actions = jrandom.uniform(jrandom.key(1), (300, env.action_dim), minval=-1.0, maxval=1.0)
    _, traj = rollout(env, params, state, actions)

    data = mujoco.MjData(env.mj_model)
    qpos_prev = np.concatenate((np.asarray(state.qpos)[None], np.asarray(traj.qpos)[:-1]))
    qvel_prev = np.concatenate((np.asarray(state.qvel)[None], np.asarray(traj.qvel)[:-1]))
    n_contact_steps = 0
    for qpos, qvel, action, mjx_qpos, mjx_qvel in zip(qpos_prev, qvel_prev, np.asarray(actions),
                                                      np.asarray(traj.qpos), np.asarray(traj.qvel)):
        mujoco.mj_resetData(env.mj_model, data)
        data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, action
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.mj_model, data)
        n_contact_steps += data.ncon > 0
        np.testing.assert_allclose(mjx_qpos, data.qpos, atol=1e-8)
        np.testing.assert_allclose(mjx_qvel, data.qvel, atol=1e-8)
    assert n_contact_steps > 100  # the comparison has to cover the contact dynamics


@pytest.mark.parametrize("env_cls", ENVS)
def test_unactuated_robot_settles(env_cls):
    """Without actuation the robot falls and comes to rest, so the dynamics are dissipative and not chaotic"""
    env = env_cls()
    params = env.default_params.replace(terminate_when_unhealthy=False)
    _, state = env.reset(params, jrandom.key(0))
    zero = jnp.zeros(env.action_dim)

    final_state, _ = rollout(env, params, state, jnp.zeros((2000, env.action_dim)))
    assert float(jnp.max(jnp.abs(final_state.qvel))) < 1e-3

    def step_fn(x):
        return jnp.concatenate(env._physics_step(x[:env.nq], x[env.nq:], zero))

    exponents = lyapunov_spectrum(step_fn, jnp.concatenate((final_state.qpos, final_state.qvel)),
                                  n_steps=1000, n_transient=0, n_exponents=2) / env.dt
    assert float(exponents[0]) < 0.1  # rest state, forward translation gives the zero exponent


@pytest.mark.parametrize("env_cls", ENVS)
def test_periodic_actuation_is_chaotic(env_cls):
    """
    Sinusoidal joint torques give a positive leading Lyapunov exponent, 3-4 /s for both robots with CPU MuJoCo. The
    forcing phase is appended to the state to make the map autonomous, which only adds a neutral direction.
    """
    env = env_cls()
    params = env.default_params.replace(terminate_when_unhealthy=False)
    _, state = env.reset(params, jrandom.key(0))
    phase_offsets = jrandom.uniform(jrandom.key(1), (env.action_dim,), maxval=2 * jnp.pi)
    d_phase = 2 * jnp.pi * FORCING_FREQUENCY * env.dt

    def step_fn(x):
        qpos, qvel, phase = x[:env.nq], x[env.nq:-1], x[-1]
        qpos, qvel = env._physics_step(qpos, qvel, 0.5 * jnp.sin(phase + phase_offsets))
        return jnp.concatenate((qpos, qvel, (phase + d_phase)[None]))

    x0 = jnp.concatenate((state.qpos, state.qvel, jnp.zeros(1)))
    exponents = lyapunov_spectrum(step_fn, x0, n_steps=4000, n_transient=500, n_exponents=2) / env.dt
    assert float(exponents[0]) > 1.0
