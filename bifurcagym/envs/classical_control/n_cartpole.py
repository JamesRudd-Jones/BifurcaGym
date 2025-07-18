"""
Based off: https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Any, Dict, Tuple, Union
import chex
from flax import struct
from bifurcagym.envs import base_env
from bifurcagym import spaces


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    thetas: jnp.ndarray
    theta_dots: jnp.ndarray
    time: int


class NCartPoleCSDA(base_env.BaseEnvironment):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.num_poles: int = 2

        self.periodic_dim = jnp.concatenate((jnp.zeros(2,), jnp.tile(jnp.array((1, 0)), self.num_poles)))

        self.gravity: float = 9.82
        self.mass_cart: float = 0.5  # 1.0
        self.mass_poles: chex.Array = jnp.tile(jnp.array((0.5,)), self.num_poles)
        self.mass_total: float = self.mass_cart + float(jnp.sum(self.mass_poles))
        self.lengths: chex.Array = jnp.tile(jnp.array((0.6,)), self.num_poles)
        self.length_total: float = float(jnp.sum(self.lengths))
        self.mu: float = 0.1  # friction coefficient

        self.action_array: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))
        self.force_mag: float = 10.0

        self.horizon: int = 25
        self.dt: float = 0.1

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        costhetas = jnp.cos(state.thetas)
        sinthetas = jnp.sin(state.thetas)

        xdot_term1 = -2 * jnp.sum(self.mass_poles * self.lengths * (state.theta_dots ** 2) * sinthetas)
        xdot_term2 = 3 * self.gravity * jnp.sum(self.mass_poles * sinthetas * costhetas)
        xdot_term3 = 4 * (action - self.mu * state.x_dot)
        xdot_denom = 4 * self.mass_total - 3 * jnp.sum(self.mass_poles * costhetas ** 2)

        xdot_update = (xdot_term1 + xdot_term2 + xdot_term3) / xdot_denom

        thetadot_term1 = -3 * self.mass_poles * self.lengths * (state.theta_dots ** 2) * sinthetas * costhetas
        thetadot_term2 = 6 * self.mass_total * self.gravity * sinthetas
        thetadot_term3 = 6 * (action - self.mu * state.x_dot) * costhetas
        thetadot_denom = 4 * self.lengths * self.mass_total - 3 * self.mass_poles * self.lengths * costhetas ** 2

        thetadot_update = (thetadot_term1 + thetadot_term2 + thetadot_term3) / thetadot_denom

        x = state.x + state.x_dot * self.dt
        unnorm_thetas = state.thetas + state.theta_dots * self.dt
        thetas = jax.vmap(self._angle_normalise)(unnorm_thetas)
        x_dot = state.x_dot + xdot_update * self.dt
        theta_dots = state.theta_dots + thetadot_update * self.dt

        # delta_s = jnp.array((x, x_dot, unnorm_thetas, theta_dots)) - self.get_obs(state)
        # TODO check why this is unnorm theta

        # Update state dict and evaluate termination conditions
        new_state = EnvState(x=x,
                             x_dot=x_dot,
                             thetas=thetas,
                             theta_dots=theta_dots,
                             time=state.time + 1)

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {"discount": self.discount(new_state)})

    @staticmethod
    def _angle_normalise(x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    @staticmethod
    def _get_coords(theta, length):
        return -length * jnp.sin(theta), length * jnp.cos(theta)

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
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

    def reward_function(self,  # TODO check this
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> chex.Array:
        goal = jnp.array([0.0, self.length_total])
        pendulum_x, pendulum_y = jax.vmap(self._get_coords)(state_tp1.thetas, self.lengths)
        position = jnp.array([state_tp1.x + float(jnp.sum(pendulum_x)), float(jnp.sum(pendulum_y))])
        squared_distance = jnp.sum((position - goal) ** 2)
        squared_sigma = 0.25 ** 2
        costs = 1 - jnp.exp(-0.5 * squared_distance / squared_sigma)

        return -costs

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action.squeeze()] * self.force_mag / 4
        # TODO need this 4 divisor to work for discrete actions

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:  # TODO check this
        # TODO if self.use_trig then it is the below
        # return jnp.array([state.x, state.x_dot, jnp.sin(state.theta), jnp.cos(state.theta), state.theta_dot])
        # Otherwise
        return jnp.concatenate((jnp.expand_dims(state.x, axis=0),
                                jnp.expand_dims(state.x_dot, axis=0),
                                state.thetas,
                                state.theta_dots))

    def get_state(self, obs: chex.Array) -> EnvState:  # TODO check this
        return EnvState(x=obs[0],
                        x_dot=obs[1],
                        thetas=jax.vmap(self._angle_normalise)(obs[2:self.num_poles+2]),
                        theta_dots=obs[self.num_poles+2:],
                        time=-1)


    def is_done(self, state: EnvState) -> chex.Array:
        return jnp.array(False)

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f"{self.name}-NumPoles={self.num_poles}")
        screen_lim = float(jnp.sum(self.lengths))
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
            pendulum_x, pendulum_y = jax.vmap(self._get_coords)(trajectory_state.thetas[frame], self.lengths)
            cart_center_x = trajectory_state.x[frame]

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
                                       frames=trajectory_state.time.shape[0],
                                       interval=self.dt * 1000,  # Convert dt to milliseconds
                                       blit=True
                                       )
        anim.save(f"{file_path}_{self.name}-NumPoles={self.num_poles}..gif")
        plt.close()

    @property
    def name(self) -> str:
        return "NCartPole-v0"

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self) -> spaces.Box:
        high = jnp.concatenate((jnp.array((10.0, 10.0)),
                                jnp.tile(jnp.array((3.14159,)), self.num_poles),
                                jnp.tile(jnp.array((25.0,)), self.num_poles)))
        return spaces.Box(-high, high, (2 + self.num_poles,), dtype=jnp.float32)


class NCartPoleCSCA(NCartPoleCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -1, 1).squeeze() * self.force_mag

    def action_space(self) -> spaces.Box:
        return spaces.Box(-1, 1, shape=(1,))