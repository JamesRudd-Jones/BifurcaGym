"""
Based off: https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Any, Dict, Tuple
import chex
from flax import struct
from bifurcagym.envs import base_env, utils
from bifurcagym import spaces
from pygments.styles import default


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    thetas: jnp.ndarray
    theta_dots: jnp.ndarray


@struct.dataclass
class EnvParams:
    gravity: float = 9.82
    mass_cart: float = 0.5  # 1.0
    mu: float = 0.1  # friction coefficient
    mass_poles: float = 0.5
    lengths: float = 0.6
    force_mag: float = 10.0

    x_threshold: float = 2.0
    maximum_x_threshold: float = struct.field(False, default=2.0)  # maximum to ensure correct scaling


class NCartPoleCSDA(base_env.BaseEnvironment):
    def __init__(self, num_poles: int = 2, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = 0.1
        self.horizon: int = 25
        self.max_steps_in_ep: int = 500
        self.num_poles: int = num_poles
        self.substeps: int = 2

        self.periodic_dim: chex.Array = jnp.concatenate((jnp.zeros(2, ), jnp.tile(jnp.array((1, 0)), self.num_poles)))

        self.action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))

        self.requires_float64: bool = True

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def mass_poles(self, params: EnvParams) -> chex.Array:
        return jnp.ones(self.num_poles) * params.mass_poles

    def lengths(self, params) -> chex.Array:
        return jnp.ones(self.num_poles) * params.lengths

    def mass_total(self, params: EnvParams) -> chex.Numeric:
        return params.mass_cart + jnp.sum(self.mass_poles(params))

    def length_total(self, params) -> chex.Numeric:
        return jnp.sum(self.lengths(params))

    def step_env(self,
                 input_action: chex.Numeric,
                 state: EnvState,
                 params: EnvParams,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action, params)

        q = jnp.concatenate([jnp.array([state.x]), state.thetas])
        q_dot = jnp.concatenate([jnp.array([state.x_dot]), state.theta_dots])
        x0 = jnp.concatenate([q, q_dot])

        # Integrate forward using precise physics
        xf = utils.integrate_ode(self.dynamics_eom, state.time * self.dt, x0, action, self.dt, self.substeps, params)

        n_dof = self.num_poles + 1
        new_q = xf[:n_dof]
        new_q_dot = xf[n_dof:]

        new_state = EnvState(x=new_q[0],
                             x_dot=new_q_dot[0],
                             thetas=jax.vmap(self._angle_normalise)(new_q[1:]),
                             theta_dots=new_q_dot[1:],
                             time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {"discount": self.discount(done)})

    def kinetic_energy(self, q: chex.Array, q_dot: chex.Array, params) -> chex.Array:
        cart_x_dot = q_dot[0]
        thetas = q[1:]
        theta_dots = q_dot[1:]

        # Cart kinetic energy
        K_cart = 0.5 * params.mass_cart * cart_x_dot ** 2

        # Absolute angles and angular velocities
        phi = jnp.cumsum(thetas)
        phi_dot = jnp.cumsum(theta_dots)

        # Expand scalar length/mass to arrays for N poles
        L = jnp.ones(self.num_poles) * params.lengths
        m = jnp.ones(self.num_poles) * params.mass_poles

        # Joint velocities (base joint moves with the cart in X)
        v_joint_x = jnp.concatenate([jnp.array([cart_x_dot]), cart_x_dot + jnp.cumsum(L * phi_dot * jnp.cos(phi))[:-1]])
        v_joint_y = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(-L * phi_dot * jnp.sin(phi))[:-1]])

        # Centre of mass velocities for each uniform rod
        v_com_x = v_joint_x + (L / 2) * phi_dot * jnp.cos(phi)
        v_com_y = v_joint_y - (L / 2) * phi_dot * jnp.sin(phi)

        v_sq = v_com_x ** 2 + v_com_y ** 2
        I = (1.0 / 12.0) * m * L ** 2

        K_poles = jnp.sum(0.5 * m * v_sq + 0.5 * I * phi_dot ** 2)
        return K_cart + K_poles

    def potential_energy(self, q: chex.Array, params) -> chex.Array:
        thetas = q[1:]
        phi = jnp.cumsum(thetas)

        L = jnp.ones(self.num_poles) * params.lengths
        m = jnp.ones(self.num_poles) * params.mass_poles

        # Joint heights (cart is at y=0, standard setup balances upward +y)
        joint_y = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(L * jnp.cos(phi))[:-1]])
        com_y = joint_y + (L / 2) * jnp.cos(phi)

        V = jnp.sum(m * params.gravity * com_y)
        return V

    def compute_bias(self, q: chex.Array, q_dot: chex.Array, params) -> chex.Array:
        G = jax.grad(self.potential_energy, argnums=0)(q, params)
        dK_dq = jax.grad(self.kinetic_energy, argnums=0)(q, q_dot, params)

        def momentum_func(q_val):
            return jax.grad(self.kinetic_energy, argnums=1)(q_val, q_dot, params)

        _, M_dot_q_dot = jax.jvp(momentum_func, (q,), (q_dot,))
        return M_dot_q_dot - dK_dq + G

    def dynamics_eom(self, t: float, x: chex.Array, u: chex.Array, params) -> chex.Array:
        n_dof = self.num_poles + 1  # cart + N poles
        q = x[:n_dof]
        q_dot = x[n_dof:]

        # Apply control force (action) and cart friction (-mu * cart_velocity)
        force = jnp.squeeze(u) - params.mu * q_dot[0]
        tau = jnp.zeros(n_dof).at[0].set(force)

        # Autodiff Mass Matrix and Bias
        M = jax.hessian(self.kinetic_energy, argnums=1)(q, q_dot, params)
        bias = self.compute_bias(q, q_dot, params)

        # Solve M * q_ddot = tau - bias
        q_ddot = jnp.linalg.solve(M, tau - bias)

        return jnp.concatenate([q_dot, q_ddot])

    @staticmethod
    def _angle_normalise(x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    @staticmethod
    def _get_coords(theta, length):
        return -length * jnp.sin(theta), length * jnp.cos(theta)

    # @partial(jax.jit, static_argnums=(0, 1))
    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        loc = jnp.array([0.0, 0.0])
        scale = jnp.array([0.02, 0.02])
        init_state_cart = jrandom.normal(key, shape=(2,)) * scale + loc

        key, _key = jrandom.split(key)
        init_state_thetas = jrandom.normal(_key, shape=(self.num_poles,)) * 0.02 + jnp.pi

        key, _key = jrandom.split(key)
        init_state_theta_dots = jrandom.normal(_key, shape=(self.num_poles,)) * 0.02 + 0.0

        state = EnvState(x=init_state_cart[0],
                         x_dot=init_state_cart[1],
                         thetas=jax.vmap(self._angle_normalise)(init_state_thetas),
                         theta_dots=init_state_theta_dots,
                         time=0)
        return self.get_obs(state), state

    def reward_and_done_function(self,  # TODO check this
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        goal = jnp.array([0.0, self.length_total(params)])

        abs_angles = jnp.cumsum(state_tp1.thetas)
        tip_x = state_tp1.x + jnp.sum(params.lengths * jnp.sin(abs_angles))
        tip_y = jnp.sum(params.lengths * jnp.cos(abs_angles))
        position = jnp.array([tip_x, tip_y])

        squared_distance = jnp.sum((position - goal) ** 2)
        squared_sigma = 0.25 ** 2
        costs = 1 - jnp.exp(-0.5 * squared_distance / squared_sigma)

        out_of_bounds = jnp.logical_or(state_tp1.x < -params.x_threshold, state_tp1.x > params.x_threshold)
        time_limit = state_tp1.time >= self.max_steps_in_ep
        fin_done = jnp.logical_or(out_of_bounds, time_limit)

        return -costs, fin_done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.force_mag / 4
        # TODO need this 4 divisor to work for discrete actions

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:  # TODO check this
        # TODO if self.use_trig then it is the below
        # return jnp.array([state.x, state.x_dot, jnp.sin(state.theta), jnp.cos(state.theta), state.theta_dot])
        # Otherwise
        return jnp.concatenate((jnp.expand_dims(state.x, axis=0),
                                jnp.expand_dims(state.x_dot, axis=0),
                                state.thetas,
                                state.theta_dots))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:  # TODO check this
        return EnvState(x=obs[0],
                        x_dot=obs[1],
                        thetas=jax.vmap(self._angle_normalise)(obs[2:self.num_poles+2]),
                        theta_dots=obs[self.num_poles+2:],
                        time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        thetas = np.asarray(trajectory_state.thetas)
        xs = np.asarray(trajectory_state.x)
        lengths = np.asarray(jax.vmap(self.lengths)(params))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f"{self.name}-NumPoles={self.num_poles}")
        screen_lim = float(np.sum(np.max(lengths, axis=0)))  # TODO should maximum over the batch and then sum all length options
        ax.set_xlim(-screen_lim * 3, screen_lim * 3)
        ax.set_xlabel("X")
        ax.set_ylim(-screen_lim* 1.2, screen_lim * 1.2)
        # ax.set_ylabel("Y")
        ax.set_aspect('equal')
        ax.grid(True)

        line, = ax.plot([], [], 'k-', lw=3, zorder=3)
        dot, = ax.plot([], [], color="r", marker="o", markersize=8, zorder=4, label='Current State')

        cart_width = 0.4
        cart_height = 0.2
        cart = plt.Rectangle((0, -cart_height / 2), cart_width, cart_height, fc='blue', ec='black', zorder=2)
        ax.add_patch(cart)

        anchor, = ax.plot([], [], 'ko', markersize=6)  # Anchor point where the pendulum is attached to the cart

        def update(frame):
            pendulum_x, pendulum_y = jax.vmap(self._get_coords)(thetas[frame], lengths[frame])
            cart_center_x = xs[frame]

            # The pendulum's top is attached to the cart
            cumulative_x = jnp.cumsum(jnp.concatenate((jnp.ones(1,) * cart_center_x, pendulum_x)))
            cumulative_y = jnp.cumsum(jnp.concatenate((jnp.zeros(1,), pendulum_y)))

            line.set_data(cumulative_x, cumulative_y)
            dot.set_data(jnp.expand_dims(cumulative_x[-1], axis=0),
                         jnp.expand_dims(cumulative_y[-1], axis=0))

            # Update cart position (x, y) is bottom-left corner for Rectangle
            cart.set_xy((cart_center_x - cart_width / 2, -cart_height / 2))

            # Update anchor point
            anchor.set_data([cart_center_x], [0])

            return line, dot, cart

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=thetas.shape[0],
                                       interval=self.dt * 1000,  # Convert dt to milliseconds
                                       blit=True
                                       )
        anim.save(f"{file_path}_{self.name}-NumPoles={self.num_poles}..gif")
        plt.close()

    @property
    def name(self) -> str:
        return "NCartPole-v0"

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        high = jnp.concatenate((jnp.array((10.0, 10.0)),
                                jnp.tile(jnp.array((3.14159,)), self.num_poles),
                                jnp.tile(jnp.array((25.0,)), self.num_poles)))
        return spaces.Box(-high, high, (2 + self.num_poles * 2,), dtype=jnp.float32)


class NCartPoleCSCA(NCartPoleCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -1, 1).squeeze() * params.force_mag

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-1, 1, shape=(1,))