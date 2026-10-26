"""
Physical validity of the classical control envs.

Each env's accelerations are compared against an independent Euler-Lagrange model built from forward kinematics of
the centres of mass, and energy / momentum are checked along the env's own uncontrolled rollouts.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from bifurcagym.envs.classical_control import acrobot, cartpole, n_cartpole, n_pendulum, pendulum
from physics_utils import lagrangian_accel, random_states, rigid_body_energies


pytestmark = pytest.mark.physics

NUM_STATES = 32


# # # Independent models, conventions are taken from each env's docstrings and reset/reward functions
########################################################################################################################
def n_pendulum_model(n_links, params):
    """Relative joint angles, theta = 0 hanging down, uniform rods, torque at the base joint"""
    m, length, g = params.mass, params.length, params.gravity

    def com_fn(q):
        coms = []
        joint = jnp.zeros(2)
        phi = 0.0
        for i in range(n_links):
            phi = phi + q[i]
            direction = jnp.array((jnp.sin(phi), -jnp.cos(phi)))
            coms.append(joint + 0.5 * length * direction)
            joint = joint + length * direction
        return jnp.stack(coms)

    kinetic, potential = rigid_body_energies(com_fn,
                                             lambda q_dot: jnp.cumsum(q_dot),
                                             jnp.full(n_links, m),
                                             jnp.full(n_links, m * length ** 2 / 12),
                                             g)

    def gen_force(q, q_dot, u):
        return jnp.zeros(n_links).at[0].set(u)

    return kinetic, potential, gen_force


def n_cartpole_model(num_poles, params):
    """Cart position then relative pole angles, theta = 0 upright, uniform rods, viscous cart friction"""
    m, length, g = params.mass_poles, params.lengths, params.gravity

    def com_fn(q):
        coms = [jnp.array((q[0], 0.0))]
        joint = jnp.array((q[0], 0.0))
        phi = 0.0
        for i in range(num_poles):
            phi = phi + q[i + 1]
            direction = jnp.array((jnp.sin(phi), jnp.cos(phi)))
            coms.append(joint + 0.5 * length * direction)
            joint = joint + length * direction
        return jnp.stack(coms)

    kinetic, potential = rigid_body_energies(com_fn,
                                             lambda q_dot: jnp.concatenate((jnp.zeros(1), jnp.cumsum(q_dot[1:]))),
                                             jnp.concatenate((jnp.array((params.mass_cart,)), jnp.full(num_poles, m))),
                                             jnp.concatenate((jnp.zeros(1), jnp.full(num_poles, m * length ** 2 / 12))),
                                             g)

    def gen_force(q, q_dot, u):
        return jnp.zeros(num_poles + 1).at[0].set(u - params.mu * q_dot[0])

    return kinetic, potential, gen_force


def cartpole_model(params):
    """theta = 0 upright with the pole leaning to -x for positive theta (PILCO convention)"""
    m, length = params.mass_pole, params.length

    def com_fn(q):
        return jnp.stack((jnp.array((q[0], 0.0)),
                          jnp.array((q[0] - 0.5 * length * jnp.sin(q[1]), 0.5 * length * jnp.cos(q[1])))))

    kinetic, potential = rigid_body_energies(com_fn,
                                             lambda q_dot: jnp.array((0.0, q_dot[1])),
                                             jnp.array((params.mass_cart, m)),
                                             jnp.array((0.0, m * length ** 2 / 12)),
                                             params.gravity)

    def gen_force(q, q_dot, u):
        return jnp.array((u - params.mu * q_dot[0], 0.0))

    return kinetic, potential, gen_force


def pendulum_model(params):
    """theta = 0 upright, uniform rod pivoting about one end"""
    m, length = params.mass, params.length

    def com_fn(q):
        return (0.5 * length * jnp.array((jnp.sin(q[0]), jnp.cos(q[0]))))[None]

    kinetic, potential = rigid_body_energies(com_fn,
                                             lambda q_dot: q_dot,
                                             jnp.array((m,)),
                                             jnp.array((m * length ** 2 / 12,)),
                                             params.gravity)

    def gen_force(q, q_dot, u):
        return jnp.array((u,))

    return kinetic, potential, gen_force


def double_link_model(params, actuated_joint):
    """Spong acrobot/pendubot, theta_1 from hanging down, theta_2 relative, inertia about each centre of mass"""
    gravity = 9.8

    def com_fn(q):
        dir_1 = jnp.array((jnp.sin(q[0]), -jnp.cos(q[0])))
        dir_2 = jnp.array((jnp.sin(q[0] + q[1]), -jnp.cos(q[0] + q[1])))
        return jnp.stack((params.com_pos_1 * dir_1,
                          params.length_1 * dir_1 + params.com_pos_2 * dir_2))

    kinetic, potential = rigid_body_energies(com_fn,
                                             lambda q_dot: jnp.array((q_dot[0], q_dot[0] + q_dot[1])),
                                             jnp.array((params.mass_1, params.mass_2)),
                                             jnp.array((params.moi, params.moi)),
                                             gravity)

    def gen_force(q, q_dot, u):
        return jnp.zeros(2).at[actuated_joint].set(u)

    return kinetic, potential, gen_force


# # # Env accelerations q_ddot(q, q_dot, u)
########################################################################################################################
def n_pendulum_case(n_links):
    env = n_pendulum.NPendulumCSCA(n_links=n_links)
    params = env.default_params

    def env_accel(q, q_dot, u):
        return env.dynamics_eom(0.0, jnp.concatenate((q, q_dot)), jnp.array((u,)), params)[n_links:]

    return env, params, n_pendulum_model(n_links, params), env_accel, n_links


def n_cartpole_case(num_poles):
    env = n_cartpole.NCartPoleCSCA(num_poles=num_poles)
    params = env.default_params

    def env_accel(q, q_dot, u):
        return env.dynamics_eom(0.0, jnp.concatenate((q, q_dot)), jnp.array((u,)), params)[num_poles + 1:]

    return env, params, n_cartpole_model(num_poles, params), env_accel, num_poles + 1


def double_link_case(env_cls, actuated_joint):
    env = env_cls()
    params = env.default_params

    def env_accel(q, q_dot, u):
        return env._dsdt(0.0, jnp.concatenate((q, q_dot)), u, params)[2:]

    return env, params, double_link_model(params, actuated_joint), env_accel, 2


def cartpole_case():
    env = cartpole.CartPoleCSCA()
    params = env.default_params

    def env_accel(q, q_dot, u):
        return env._dsdt(0.0, jnp.concatenate((q, q_dot)), u, params)[2:]

    return env, params, cartpole_model(params), env_accel, 2


def pendulum_case():
    env = pendulum.PendulumCSCA()
    params = env.default_params.replace(max_speed=1e6)  # avoid the velocity clip

    def env_accel(q, q_dot, u):
        state = pendulum.EnvState(theta=q[0], theta_dot=q_dot[0], time=0)
        _, _, new_state, _, _, _ = env.step(jnp.array((u,)), state, params, jrandom.key(0))
        return jnp.array((new_state.theta_dot - q_dot[0],)) / env.dt

    return env, params, pendulum_model(params), env_accel, 1


CASES = {"NPendulum-1": lambda: n_pendulum_case(1),
         "NPendulum-2": lambda: n_pendulum_case(2),
         "NPendulum-3": lambda: n_pendulum_case(3),
         "NCartPole-1": lambda: n_cartpole_case(1),
         "NCartPole-2": lambda: n_cartpole_case(2),
         "Acrobot": lambda: double_link_case(acrobot.AcrobotCSCA, actuated_joint=1),
         "Pendubot": lambda: double_link_case(acrobot.PendubotCSCA, actuated_joint=0),
         "CartPole": cartpole_case,
         "Pendulum": pendulum_case,
         }


@pytest.mark.parametrize("case", CASES.keys())
def test_accelerations_match_euler_lagrange(case):
    """Env equations of motion match an independent Lagrangian model, including the control input"""
    env, params, (kinetic, potential, gen_force), env_accel, dof = CASES[case]()
    lagrangian = lambda q, q_dot: kinetic(q, q_dot) - potential(q)

    key_q, key_qd, key_u = jrandom.split(jrandom.key(0), 3)
    qs = random_states(key_q, NUM_STATES, -jnp.pi * jnp.ones(dof), jnp.pi * jnp.ones(dof))
    q_dots = random_states(key_qd, NUM_STATES, -2.0 * jnp.ones(dof), 2.0 * jnp.ones(dof))
    us = random_states(key_u, NUM_STATES, -1.0, 1.0)

    expected = jax.vmap(lambda q, qd, u: lagrangian_accel(lagrangian, q, qd, gen_force(q, qd, u)))(qs, q_dots, us)
    actual = jax.vmap(env_accel)(qs, q_dots, us)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


# # # Conservation along the env's own rollouts
########################################################################################################################
def rollout_energy(env, params, state, kinetic, potential, to_q, n_steps):
    action = jnp.zeros(env.action_space(params).shape)

    def body(state, key):
        _, _, new_state, _, _, _ = env.step(action, state, params, key)
        q, q_dot = to_q(new_state)
        return new_state, kinetic(q, q_dot) + potential(q)

    q0, q_dot0 = to_q(state)
    _, energies = jax.lax.scan(body, state, jrandom.split(jrandom.key(1), n_steps))
    return kinetic(q0, q_dot0) + potential(q0), energies


ENERGY_CASES = {
    # case: (state builder, state -> (q, q_dot), params changes, relative drift tolerance over 1000 env steps)
    "NPendulum-2": (lambda env: n_pendulum.EnvState(theta=jnp.array((0.8, -0.5)), theta_dot=jnp.array((0.3, 0.1)), time=0),
                    lambda s: (s.theta, s.theta_dot), dict(max_speed=1e6), 1e-3),
    "NPendulum-3": (lambda env: n_pendulum.EnvState(theta=jnp.array((0.6, -0.4, 0.3)), theta_dot=jnp.zeros(3), time=0),
                    lambda s: (s.theta, s.theta_dot), dict(max_speed=1e6), 1e-3),
    "NCartPole-1": (lambda env: n_cartpole.EnvState(x=jnp.array(0.0), x_dot=jnp.array(0.1), thetas=jnp.array((2.5,)),
                                                    theta_dots=jnp.array((0.2,)), time=0),
                    lambda s: (jnp.concatenate((s.x[None], s.thetas)), jnp.concatenate((s.x_dot[None], s.theta_dots))),
                    dict(mu=0.0), 2e-2),
    "NCartPole-2": (lambda env: n_cartpole.EnvState(x=jnp.array(0.0), x_dot=jnp.array(0.1), thetas=jnp.array((2.8, 0.3)),
                                                    theta_dots=jnp.array((0.2, -0.1)), time=0),
                    lambda s: (jnp.concatenate((s.x[None], s.thetas)), jnp.concatenate((s.x_dot[None], s.theta_dots))),
                    dict(mu=0.0), 2e-2),
    "CartPole": (lambda env: cartpole.EnvState(x=jnp.array(0.0), x_dot=jnp.array(0.1), theta=jnp.array(2.5),
                                               theta_dot=jnp.array(0.2), time=0),
                 lambda s: (jnp.array((s.x, s.theta)), jnp.array((s.x_dot, s.theta_dot))), dict(mu=0.0), 2e-2),
    "Acrobot": (lambda env: acrobot.EnvState(joint_angle_1=jnp.array(0.5), joint_angle_2=jnp.array(-0.3),
                                             vel_1=jnp.array(0.2), vel_2=jnp.array(0.1), time=0),
                lambda s: (jnp.array((s.joint_angle_1, s.joint_angle_2)), jnp.array((s.vel_1, s.vel_2))), {}, 5e-2),
    "Pendubot": (lambda env: acrobot.EnvState(joint_angle_1=jnp.array(0.5), joint_angle_2=jnp.array(-0.3),
                                              vel_1=jnp.array(0.2), vel_2=jnp.array(0.1), time=0),
                 lambda s: (jnp.array((s.joint_angle_1, s.joint_angle_2)), jnp.array((s.vel_1, s.vel_2))), {}, 5e-2),
}


ENERGY_MODELS = {"NPendulum-2": lambda params: n_pendulum_model(2, params),
                 "NPendulum-3": lambda params: n_pendulum_model(3, params),
                 "NCartPole-1": lambda params: n_cartpole_model(1, params),
                 "NCartPole-2": lambda params: n_cartpole_model(2, params),
                 "CartPole": lambda params: cartpole_model(params),
                 "Acrobot": lambda params: double_link_model(params, 1),
                 "Pendubot": lambda params: double_link_model(params, 0),
                 }


def refine(env, factor):
    """Shrinks the integration step by factor, returns how many more env steps cover the same time"""
    if hasattr(env, "substeps"):
        env.substeps *= factor
        return 1
    env.dt /= factor
    return factor


@pytest.mark.parametrize("case", ENERGY_CASES.keys())
def test_energy_conserved_without_friction_or_control(case):
    """
    Energy drift at the env's own time step stays within tolerance, and vanishes at the RK4 rate as the step is
    refined, which a wrong equation of motion could not do
    """
    build_state, to_q, param_changes, rtol = ENERGY_CASES[case]
    drifts = []
    for factor in [1, 4]:
        env, params, *_ = CASES[case]()
        params = params.replace(**param_changes)
        kinetic, potential, _ = ENERGY_MODELS[case](params)
        steps_multiplier = refine(env, factor)
        e0, energies = rollout_energy(env, params, build_state(env), kinetic, potential, to_q,
                                      n_steps=1000 * steps_multiplier)
        assert jnp.all(jnp.isfinite(energies))
        drifts.append(float(jnp.max(jnp.abs(energies - e0)) / jnp.maximum(jnp.abs(e0), 1.0)))

    assert drifts[0] < rtol, f"{case} energy drifted by {drifts[0]:.2e} (relative) at the env time step"
    assert drifts[1] < drifts[0] / 4 ** 3.5, f"{case} energy drift does not converge at 4th order: {drifts}"


def test_n_cartpole_friction_dissipates_energy():
    env = n_cartpole.NCartPoleCSCA(num_poles=2)
    params = env.default_params.replace(mu=0.5)
    kinetic, potential, _ = n_cartpole_model(2, params)
    build_state, to_q, _, _ = ENERGY_CASES["NCartPole-2"]

    e0, energies = rollout_energy(env, params, build_state(env), kinetic, potential, to_q, n_steps=500)
    energies = jnp.concatenate((e0[None], energies))

    assert jnp.all(jnp.diff(energies) <= 1e-9), "Energy increased despite friction and no control"
    assert energies[-1] < energies[0]


@pytest.mark.parametrize("num_poles", [1, 2])
def test_n_cartpole_horizontal_momentum_conserved(num_poles):
    """With no friction and no applied force there is no external horizontal force on the cart and poles"""
    build_state, to_q, _, _ = ENERGY_CASES[f"NCartPole-{num_poles}"]

    def momentum_drift(substep_factor):
        # New env each time as step is jitted with the env as a static argument
        env = n_cartpole.NCartPoleCSCA(num_poles=num_poles)
        env.substeps *= substep_factor
        params = env.default_params.replace(mu=0.0)
        kinetic, _, _ = n_cartpole_model(num_poles, params)
        action = jnp.zeros(env.action_space(params).shape)

        def momentum(state):
            q, q_dot = to_q(state)
            return jax.grad(kinetic, argnums=1)(q, q_dot)[0]

        def body(state, key):
            _, _, new_state, _, _, _ = env.step(action, state, params, key)
            return new_state, momentum(new_state)

        state = build_state(env)
        _, momenta = jax.lax.scan(body, state, jrandom.split(jrandom.key(2), 1000))
        return float(jnp.max(jnp.abs(momenta - momentum(state))))

    # Momentum is nonlinear in the state so RK4 only conserves it to the integration error, refine to check it vanishes
    drift, drift_fine = momentum_drift(1), momentum_drift(4)
    assert drift < 1e-2, f"Horizontal momentum drifted by {drift:.2e}"
    assert drift_fine < drift / 4 ** 3.5, f"Momentum drift does not converge at 4th order: {drift}, {drift_fine}"


def test_pendulum_symplectic_euler_energy_bounded():
    """Semi-implicit Euler should keep the energy error bounded with no secular drift"""
    env = pendulum.PendulumCSCA()
    params = env.default_params.replace(max_speed=1e6)
    kinetic, potential, _ = pendulum_model(params)
    state = pendulum.EnvState(theta=jnp.array(2.5), theta_dot=jnp.array(0.0), time=0)  # large swing about hanging

    e0, energies = rollout_energy(env, params, state, kinetic, potential,
                                  lambda s: (s.theta[None], s.theta_dot[None]), n_steps=4000)
    rel_err = jnp.abs(energies - e0) / jnp.abs(e0)

    assert float(jnp.max(rel_err)) < 0.05
    # No secular growth: late errors no larger than early ones
    assert float(jnp.max(rel_err[-1000:])) < 1.5 * float(jnp.max(rel_err[:1000])) + 1e-6


# # # Linearisations against closed form results
########################################################################################################################
def linearised_frequencies(env, params, n_links):
    jac = jax.jacfwd(lambda x: env.dynamics_eom(0.0, x, jnp.zeros(1), params))(jnp.zeros(2 * n_links))
    eigs = np.linalg.eigvals(np.asarray(jac))
    np.testing.assert_allclose(eigs.real, 0.0, atol=1e-8)  # hanging equilibrium is a centre
    return np.sort(np.abs(eigs.imag))[::2]


def test_single_rod_pendulum_small_angle_frequency():
    env = n_pendulum.NPendulumCSCA(n_links=1)
    params = env.default_params
    omega = linearised_frequencies(env, params, 1)
    np.testing.assert_allclose(omega, np.sqrt(3 * params.gravity / (2 * params.length)), rtol=1e-8)


def test_double_rod_pendulum_normal_modes():
    """Closed form linearised double compound pendulum in absolute angles: M = mL^2[[4/3, 1/2], [1/2, 1/3]],
    K = mgL[[3/2, 0], [0, 1/2]]"""
    env = n_pendulum.NPendulumCSCA(n_links=2)
    params = env.default_params
    m, length, g = params.mass, params.length, params.gravity
    mass_matrix = m * length ** 2 * np.array(((4 / 3, 1 / 2), (1 / 2, 1 / 3)))
    stiffness = m * g * length * np.array(((3 / 2, 0.0), (0.0, 1 / 2)))
    expected = np.sort(np.sqrt(np.linalg.eigvals(np.linalg.solve(mass_matrix, stiffness)).real))

    np.testing.assert_allclose(linearised_frequencies(env, params, 2), expected, rtol=1e-8)


@pytest.mark.parametrize("num_poles", [1, 2])
def test_n_cartpole_equilibria(num_poles):
    env = n_cartpole.NCartPoleCSCA(num_poles=num_poles)
    params = env.default_params
    n_dof = num_poles + 1
    f = lambda x: env.dynamics_eom(0.0, x, jnp.zeros(1), params)

    upright = jnp.zeros(2 * n_dof)
    hanging = upright.at[1].set(jnp.pi)
    np.testing.assert_allclose(f(upright), 0.0, atol=1e-10)
    np.testing.assert_allclose(f(hanging), 0.0, atol=1e-10)

    # Upright is unstable, hanging is stable (friction damps the cart so real parts are <= 0)
    assert np.max(np.linalg.eigvals(np.asarray(jax.jacfwd(f)(upright))).real) > 0.1
    assert np.max(np.linalg.eigvals(np.asarray(jax.jacfwd(f)(hanging))).real) < 1e-8
