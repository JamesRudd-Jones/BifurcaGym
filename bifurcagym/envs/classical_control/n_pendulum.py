import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env, utils
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple
import chex


@struct.dataclass
class EnvState(base_env.EnvState):
    theta: jnp.ndarray      # shape (n_links,)
    theta_dot: jnp.ndarray  # shape (n_links,)


@struct.dataclass
class EnvParams:
    # for now links are standardised to be the same
    max_speed: float = 8.0
    maximum_max_speed: float = struct.field(False, default=8.0)  # maximum to ensure correct scaling
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0

    max_torque: float = 2.0
    maximum_max_torque: float = struct.field(False, default=2.0)  # maximum to ensure correct scaling


class NPendulumCSDA(base_env.BaseEnvironment):
    """
    N-link pendulum environment where control torque is applied only at the base joint (index 0)
    Angles are relative joint angles; absolute link orientation = cumsum(theta)
    Approximate dynamics: gravity acts on each link (via absolute orientation), torques only at base joint; inertial coupling is not modelled here
    """

    def __init__(self, n_links: int=2, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = 0.05
        self.horizon: int = 200
        self.max_steps_in_ep: int = 1000
        self.n_links:int = n_links
        self.substeps: int = 2

        self.periodic_dim: chex.Array = jnp.array((1, 0))  # TODO is this the best way?

        self.action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))

        self.requires_float64: bool = True

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(self,
                 input_action: chex.Numeric,
                 state: EnvState,
                 params: EnvParams,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action, params)

        x0 = jnp.concatenate([state.theta, state.theta_dot])

        xf = utils.integrate_ode(self.dynamics_eom, state.time * self.dt, x0, action, self.dt, self.substeps, params)

        new_theta = xf[:self.n_links]
        new_theta_dot = xf[self.n_links:]

        new_theta = self._angle_normalise(new_theta)
        new_theta_dot = jnp.clip(new_theta_dot, -params.max_speed, params.max_speed)

        new_state = EnvState(theta=new_theta, theta_dot=new_theta_dot, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    @staticmethod
    def kinetic_energy(q: chex.Array, q_dot: chex.Array, params) -> chex.Array:
        phi = jnp.cumsum(q)
        phi_dot = jnp.cumsum(q_dot)

        L = params.length
        m = params.mass

        # Velocities of joints 0 to N-1
        v_joint_x = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(L * phi_dot * jnp.cos(phi))[:-1]])
        v_joint_y = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(L * phi_dot * jnp.sin(phi))[:-1]])

        # Velocities of centre of mass (COM) for each link (assuming uniform rods)
        v_com_x = v_joint_x + (L / 2) * phi_dot * jnp.cos(phi)
        v_com_y = v_joint_y + (L / 2) * phi_dot * jnp.sin(phi)

        v_sq = v_com_x ** 2 + v_com_y ** 2
        I = (1.0 / 12.0) * m * L ** 2  # Moment of inertia for uniform rod about COM

        # K = 1/2 m v^2 + 1/2 I w^2
        K = jnp.sum(0.5 * m * v_sq + 0.5 * I * phi_dot ** 2)
        return K

    @staticmethod
    def potential_energy(q: chex.Array, params) -> chex.Array:
        phi = jnp.cumsum(q)
        L = params.length
        m = params.mass

        # Y-positions of joints 0 to N-1 (y goes down negatively)
        dy = -L * jnp.cos(phi)
        joint_y = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dy)[:-1]])

        # COM Y-positions
        com_y = joint_y - (L / 2) * jnp.cos(phi)

        # V = m * g * h
        V = jnp.sum(m * params.gravity * com_y)
        return V

    def compute_bias(self, q: chex.Array, q_dot: chex.Array, params) -> chex.Array:
        G = jax.grad(self.potential_energy, argnums=0)(q, params)
        dK_dq = jax.grad(self.kinetic_energy, argnums=0)(q, q_dot, params)

        # To get \dot{M} * \dot{q}, we take the Jacobian-vector product of momentum wrt q
        def momentum_func(q_val):
            return jax.grad(self.kinetic_energy, argnums=1)(q_val, q_dot, params)

        _, M_dot_q_dot = jax.jvp(momentum_func, (q,), (q_dot,))

        return M_dot_q_dot - dK_dq + G

    def dynamics_eom(self, t: float, x: chex.Array, u: chex.Array, params) -> chex.Array:
        n = self.n_links
        q = x[:n]
        q_dot = x[n:]

        # Apply control torque ONLY at the base joint
        tau = jnp.zeros(n).at[0].set(jnp.squeeze(u))

        # Mass matrix M(q) is the Hessian of Kinetic Energy wrt velocities
        M = jax.hessian(self.kinetic_energy, argnums=1)(q, q_dot, params)
        bias = self.compute_bias(q, q_dot, params)

        # Solve M * q_ddot = tau - bias
        q_ddot = jnp.linalg.solve(M, tau - bias)

        return jnp.concatenate([q_dot, q_ddot])

    @staticmethod
    def _angle_normalise(x: jnp.ndarray) -> jnp.ndarray:
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        high_theta = jnp.pi * 0.5
        high_thdot = 1.0
        key1, key2 = jrandom.split(key)
        theta_init = jrandom.uniform(key1, shape=(self.n_links,), minval=-high_theta, maxval=high_theta)
        thdot_init = jrandom.uniform(key2, shape=(self.n_links,), minval=-high_thdot, maxval=high_thdot)

        state = EnvState(theta=theta_init,
                         theta_dot=thdot_init,
                         time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        angle_abs_tp1 = jnp.cumsum(state_tp1.theta)
        cost_angles = jnp.sum(self._angle_normalise(angle_abs_tp1) ** 2)
        cost_vel = 0.1 * jnp.sum(state_tp1.theta_dot ** 2)
        action_vec = self.action_convert(input_action_t, params)
        cost_action = 0.001 * jnp.sum(action_vec ** 2)

        done = jnp.asarray(state_tp1.time >= self.max_steps_in_ep)

        return -(cost_angles + cost_vel + cost_action), done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Array:
        return self.action_array[action.squeeze()] * params.max_torque

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.concatenate([state.theta, state.theta_dot])

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        n = self.n_links
        return EnvState(theta=obs[:n], theta_dot=obs[n:], time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        # compute link endpoints for a trajectory: theta trajectory shape (T, n_links)
        theta_traj = np.asarray(trajectory_state.theta)      # can be (T, n) or (n,) single time
        lengths = np.asarray(params.length)

        # Ensure theta_traj is (T, n)
        if theta_traj.ndim == 1:
            theta_traj = theta_traj[np.newaxis, ...]
        T = theta_traj.shape[0]

        def endpoints_for_frame(theta_frame, length):
            # theta_frame: (n,)
            abs_angles = np.cumsum(theta_frame)  # absolute orientation of each link
            xs, ys = [], []
            x, y = 0.0, 0.0
            for ang in abs_angles:
                dx = length * np.sin(ang)
                dy = -length * np.cos(ang)
                x += dx
                y += dy
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        max_length = np.max(lengths)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        reach = max_length * self.n_links
        ax.set_xlim(-reach * 1.2, reach * 1.2)
        ax.set_ylim(-reach * 1.2, reach * 1.2)
        ax.set_aspect('equal')
        ax.grid(True)

        # initial coordinates
        xs0, ys0 = endpoints_for_frame(theta_traj[0], lengths[0])
        line, = ax.plot([0.0] + xs0.tolist(), [0.0] + ys0.tolist(), lw=3, c='k', marker='o')
        circle_patches = []
        bob_radius = 0.08
        for (x_i, y_i) in zip(xs0, ys0):
            circle = plt.Circle((float(x_i), float(y_i)), bob_radius, fc='r', zorder=3)
            ax.add_patch(circle)
            circle_patches.append(circle)

        def update(frame):
            xs, ys = endpoints_for_frame(theta_traj[frame], lengths[frame])
            line.set_data([0.0] + xs.tolist(), [0.0] + ys.tolist())
            for (circle, x_i, y_i) in zip(circle_patches, xs, ys):
                circle.set_center((float(x_i), float(y_i)))
            return (line, *circle_patches)

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=T,
                                       interval=self.dt * 1000,
                                       blit=True
                                       )
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()
    # TODO sort out render_traj as unsure it actually works as of now

    @property
    def name(self) -> str:
        return "NPendulum-v0"

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        high = jnp.concatenate([jnp.ones((self.n_links,)) * jnp.pi, jnp.ones((self.n_links,)) * params.maximum_max_speed])
        return spaces.Box(-high, high, (2 * self.n_links,), dtype=jnp.float32)


class NPendulumCSCA(NPendulumCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_torque, params.max_torque).squeeze()

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_torque, params.maximum_max_torque, shape=(1,))
