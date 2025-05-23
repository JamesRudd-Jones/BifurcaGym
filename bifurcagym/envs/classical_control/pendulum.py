import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex


@struct.dataclass
class EnvState(base_env.EnvState):
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


class PendulumCSDA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.periodic_dim: chex.Array = jnp.array((1, 0))

        self.max_speed: float = 8.0
        self.gravity: float = 10.0
        self.mass: float = 1.0
        self.length: float = 1.0

        self.action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))
        self.max_torque: float = 2.0

        self.horizon: int = 200
        self.dt: float = 0.05

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        newthdot = (state.theta_dot + (-3 * self.gravity / (2 * self.length) * jnp.sin(state.theta + jnp.pi) +
                                       3.0 / (self.mass * self.length ** 2) * action) * self.dt)
        unnorm_newth = state.theta + newthdot * self.dt
        newth = self._angle_normalise(unnorm_newth)
        newthdot = jnp.clip(newthdot, -self.max_speed, self.max_speed)

        # delta_s = jnp.array((unnorm_newth, newthdot)) - self.get_obs(state)
        # TODO check why this is unnorm_newth and that from the original

        new_state = EnvState(theta=newth, theta_dot=newthdot, time=state.time+1)

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {})

    @staticmethod
    def _angle_normalise(x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        high = jnp.array([jnp.pi, 1])
        init_state = jrandom.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = EnvState(theta=init_state[0],
                         theta_dot=init_state[1],
                         time=0)

        return self.get_obs(state), state

    def reward_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> chex.Array:
        action_t = self.action_convert(input_action_t)
        costs = self._angle_normalise(state_tp1.theta) ** 2 + 0.1 * state_tp1.theta_dot ** 2 + 0.001 * (action_t ** 2)

        return -costs

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action] * self.max_torque

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array([state.theta, state.theta_dot])

    def get_state(self, obs: chex.Array) -> EnvState:
        return EnvState(theta=obs[0], theta_dot=obs[1], time=-1)

    def is_done(self, state: EnvState) -> chex.Array:
        return jnp.array(False)

    def render_traj(self, trajectory_state: EnvState):
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
        anim.save(f"../animations/{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "Pendulum-v0"

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self) -> spaces.Box:
        high = jnp.array([jnp.pi, self.max_speed])
        return spaces.Box(-high, high, (2,), dtype=jnp.float32)


class PendulumCSCA(PendulumCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_torque, self.max_torque).squeeze()

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_torque, self.max_torque, shape=(1,))
