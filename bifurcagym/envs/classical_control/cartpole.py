"""
Based off: https://github.com/fusion-ml/trajectory-information-rl/blob/main/barl/envs/pilco_cartpole.py
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
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


class CartPoleCSDA(base_env.BaseEnvironment):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.periodic_dim = jnp.array((0, 0, 1, 0))  # TODO is this the best way?

        self.gravity: float = 9.82
        self.mass_cart: float = 0.5  # 1.0
        self.mass_pole: float = 0.5  # 0.1
        self.mass_total: float = self.mass_cart + self.mass_pole
        self.length: float = 0.6  # 0.5
        self.mass_pole_length: float = self.mass_pole * self.length
        self.mu: float = 0.1  # friction coefficient

        self.action_array: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))
        self.force_mag: float = 10.0

        self.x_threshold: float = 2.0
        self.theta_threshold: float = 12 * 2 * jnp.pi / 360

        self.horizon: int = 25
        self.dt: float = 0.1

        self.max_steps_in_episode: int = 500

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        xdot_term1 = -2 * self.mass_pole_length * (state.theta_dot ** 2) * sintheta
        xdot_term2 = 3 * self.mass_pole * self.gravity * sintheta * costheta
        xdot_term3 = 4 * action - 4 * self.mu * state.x_dot
        xdot_denom = 4 * self.mass_total - 3 * self.mass_pole * costheta ** 2

        xdot_update = (xdot_term1 + xdot_term2 + xdot_term3) / xdot_denom

        thetadot_term1 = -3 * self.mass_pole_length * (state.theta_dot ** 2) * sintheta * costheta
        thetadot_term2 = 6 * self.mass_total * self.gravity * sintheta
        theatdot_term3 = 6 * (action - self.mu * state.x_dot) * costheta
        thetadot_denom = 4 * self.length * self.mass_total - 3 * self.mass_pole_length * costheta ** 2

        thetadot_update = (thetadot_term1 + thetadot_term2 + theatdot_term3) / thetadot_denom

        # xdot_update = ((-2 * self.mass_pole_length * (state.theta_dot ** 2) * sintheta + 3 * self.mass_pole * self.gravity
        #                 * sintheta * costheta + 4 * action - 4 * self.mu * state.x_dot) /
        #                (4 * self.mass_total - 3 * self.mass_pole * costheta ** 2))
        #
        # thetadot_update = ((-3 * self.mass_pole_length * (state.theta_dot ** 2) * sintheta * costheta + 6 *
        #                    self.mass_total * self.gravity * sintheta + 6 * (action - self.mu * state.x_dot) * costheta) /
        #                    (4 * self.length * self.mass_total - 3 * self.mass_pole_length * costheta ** 2))

        x = state.x + state.x_dot * self.dt
        unnorm_theta = state.theta + state.theta_dot * self.dt
        theta = self._angle_normalise(unnorm_theta)
        x_dot = state.x_dot + xdot_update * self.dt
        theta_dot = state.theta_dot + thetadot_update * self.dt

        delta_s = jnp.array((x, x_dot, unnorm_theta, theta_dot)) - self.get_obs(state)
        # TODO check why this is unnorm theta

        # Update state dict and evaluate termination conditions
        new_state = EnvState(x=x,
                             x_dot=x_dot,
                             theta=theta,
                             theta_dot=theta_dot,
                             time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {"discount": self.discount(done)})

    @staticmethod
    def _angle_normalise(x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # loc = jnp.array([0.0, 0.0, jnp.pi, 0.0])
        # TODO the above is for swing up
        loc = jnp.array([0.0, 0.0, 0.0, 0.0])
        # TODO this is for no swing up

        scale = jnp.array([0.02, 0.02, 0.02, 0.02])
        init_state = jrandom.normal(key, shape=(4,)) * scale + loc
        state = EnvState(x=init_state[0],
                         x_dot=init_state[1],
                         theta=self._angle_normalise(init_state[2]),
                         theta_dot=init_state[3],
                         time=0)
        return self.get_obs(state), state

    def reward_and_done_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> Tuple[chex.Array, chex.Array]:
        goal = jnp.array([0.0, self.length])
        pole_x = self.length * jnp.sin(state_tp1.theta)
        pole_y = self.length * jnp.cos(state_tp1.theta)
        position = jnp.array([state_tp1.x + pole_x, pole_y])
        squared_distance = jnp.sum((position - goal) ** 2)
        squared_sigma = 0.25 ** 2
        costs = 1 - jnp.exp(-0.5 * squared_distance / squared_sigma)

        done1 = jnp.logical_or(state_tp1.x < -self.x_threshold,  # TODO state_t or state_tp1
                               state_tp1.x > self.x_threshold)

        # done2 = jnp.logical_or(state_tp1.theta < -self.theta_threshold,
        #                        state_tp1.theta > self.theta_threshold,
        #                        )
        # TODO the above is for no swingup

        done2 = False
        # TODO the above is for swingup

        done = jnp.logical_or(done1, done2)

        fin_done = jnp.logical_or(done, state_tp1.time >= self.max_steps_in_episode)

        return -costs, fin_done

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action.squeeze()] * self.force_mag / 4
        # TODO need this 4 divisor to work for discrete actions

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        # TODO if self.use_trig then it is the below
        # return jnp.array([state.x, state.x_dot, jnp.sin(state.theta), jnp.cos(state.theta), state.theta_dot])
        # Otherwise
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        return EnvState(x=obs[0], x_dot=obs[1], theta=self._angle_normalise(obs[2]), theta_dot=obs[3], time=-1)

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        def get_coords(theta):
            return -self.length * jnp.sin(theta), self.length * jnp.cos(theta)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        ax.set_xlim(-self.length * 3, self.length * 3)
        ax.set_xlabel("X")
        ax.set_ylim(-self.length * 1.2, self.length * 1.2)
        # ax.set_ylabel("Y")
        ax.set_aspect('equal')
        ax.grid(True)
        x0, y0 = get_coords(trajectory_state.theta[0])

        line, = ax.plot([], [], 'k-', lw=3, zorder=3)
        dot, = ax.plot([], [], color="r", marker="o", markersize=8, zorder=4)

        cart_width = 0.4
        cart_height = 0.2
        cart = plt.Rectangle((0, -cart_height / 2), cart_width, cart_height, fc='blue', ec='black', zorder=2)
        ax.add_patch(cart)

        anchor, = ax.plot([], [], 'ko', markersize=6)  # Anchor point where the pendulum is attached to the cart

        def update(frame):
            pendulum_x, pendulum_y = get_coords(trajectory_state.theta[frame])
            cart_center_x = trajectory_state.x[frame]

            # The pendulum's top is attached to the cart
            line.set_data([cart_center_x, cart_center_x + pendulum_x], [0, pendulum_y])
            dot.set_data([cart_center_x + pendulum_x], [pendulum_y])

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
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "CartPole-v0"

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self) -> spaces.Box:
        high = jnp.array([self.x_threshold,
                          self.x_threshold,
                          3.14159,
                          25.0])
        return spaces.Box(-high, high, (4,), dtype=jnp.float32)

    def reward_space(self) -> spaces.Box:
        return spaces.Box(-1, 0, (()), dtype=jnp.float32)


class CartPoleCSCA(CartPoleCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -1, 1).squeeze() * self.force_mag

    def action_space(self) -> spaces.Box:
        return spaces.Box(-1, 1, shape=(1,))