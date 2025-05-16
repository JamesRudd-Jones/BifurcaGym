import numpy as np
from os import path
import jax.numpy as jnp
import jax.random as jrandom
from project_name.envs import base_env
from gymnax.environments import spaces
from flax import struct
from typing import Any, Dict, Optional, Tuple, Union
import chex
import jax
import matplotlib.pyplot as plt


@struct.dataclass
class EnvState(base_env.EnvState):
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


class PendulumCSDA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.periodic_dim: chex.Array = jnp.array((1, 0))  # TODO is this the best way?

        self.max_speed: float = 8.0
        self.max_torque: float = 2.0
        self.dt: float = 0.05
        self.gravity: float = 10.0
        self.mass: float = 1.0
        self.length: float = 1.0
        self.horizon: int = 200

        self.action_array: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))

    def step_env(self,
                 input_action: Union[int, float, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        action = self._action_convert(input_action)

        newthdot = (state.theta_dot + (-3 * self.gravity / (2 * self.length) * jnp.sin(state.theta + jnp.pi) +
                                       3.0 / (self.mass * self.length ** 2) * action) * self.dt)
        unnorm_newth = state.theta + newthdot * self.dt
        newth = self._angle_normalise(unnorm_newth)
        newthdot = jnp.clip(newthdot, -self.max_speed, self.max_speed)

        costs = self._angle_normalise(newth) ** 2 + 0.1 * newthdot ** 2 + 0.001 * (action ** 2)

        delta_s = jnp.array((unnorm_newth, newthdot)) - self.get_obs(state)

        state = EnvState(theta=newth, theta_dot=newthdot, time=state.time+1)

        done = jnp.array(False)

        return (jax.lax.stop_gradient(self.get_obs(state)),
                jax.lax.stop_gradient(state),
                jnp.array(-costs),
                done,
                {"delta_obs": delta_s})

    def generative_step_env(self,
                            action: Union[int, float, chex.Array],
                            obs: chex.Array,
                            key: chex.PRNGKey,
                            ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        state = EnvState(theta=obs[0], theta_dot=obs[1], time=0)
        return self.step(action, state, key)

    def _action_convert(self, input_action):
        return self.action_array[input_action] * self.max_torque

    def _angle_normalise(self, x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def reward_func(self,
                    x_t: chex.Array,
                    x_tp1: chex.Array,
                    key: chex.PRNGKey,
                    ) -> chex.Array:
        th = x_tp1[..., 0]
        thdot = x_tp1[..., 1]
        u = x_t[..., 2]
        costs = self._angle_normalise(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        return -costs

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        high = jnp.array([jnp.pi, 1])
        init_state = jrandom.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = EnvState(theta=init_state[0],
                         theta_dot=init_state[1],
                         time=0)

        return self.get_obs(state), state

    def render_traj(self, trajectory_state: EnvState):
        """Render the pendulum's trajectory as an animation.

        Args:
            trajectory: A chex.Array of shape (n_steps, 2), where each row
                represents the state [theta, theta_dot] at a given time step.
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        def get_coords(theta):
            return -self.length * jnp.sin(theta), self.length * jnp.cos(theta)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        ax.set_xlim(-self.length * 1.2, self.length * 1.2)
        ax.set_xlabel("X")
        ax.set_ylim(-self.length * 1.2, self.length * 1.2)
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        ax.grid(True)
        x0, y0 = get_coords(trajectory_state.theta[0])
        line, = ax.plot([0, x0], [0, y0], lw=3, c='k')
        # time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        bob_radius = 0.08
        circle = ax.add_patch(plt.Circle((x0, y0), bob_radius, fc='r', zorder=3))

        def update(frame):
            x, y = get_coords(trajectory_state.theta[frame])
            line.set_data([0, x], [0, y])
            circle.set_center((x, y))
            return line, circle

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=trajectory_state.time.shape[0],
                                       interval=self.dt * 1000,  # Convert dt to milliseconds
                                       blit=True
                                       )
        anim.save(f"./animations/{self.name}.gif")
        plt.close()

    def get_obs(self, state: EnvState, key=None) -> chex.Array:
        return jnp.array([state.theta, state.theta_dot])

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pendulum-v0"

    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_array))

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([jnp.pi, self.max_speed])
        return spaces.Box(-high, high, (2,), dtype=jnp.float32)

    # TODO add in state space


class PendulumCSCA(PendulumCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def _action_convert(self, input_action):
        return jnp.clip(input_action, -self.max_torque, self.max_torque)[0]

    def action_space(self) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-self.max_torque, self.max_torque, shape=(1,))
