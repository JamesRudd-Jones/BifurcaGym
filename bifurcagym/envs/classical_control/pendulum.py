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


@struct.dataclass
class EnvParams:
    action_array: jnp.ndarray = struct.field(pytree_node=False, default=(jnp.array((0.0, 1.0, -1.0))))
    dt: float = struct.field(pytree_node=False, default=0.05)
    horizon: int = struct.field(pytree_node=False, default=200)
    max_steps_in_ep: int = struct.field(pytree_node=False, default=1000)
    periodic_dim: jnp.ndarray = struct.field(pytree_node=False, default=jnp.array((1, 0)))  # TODO is this the best way?

    max_speed: float = 8.0
    maximum_max_speed: float = struct.field(pytree_node=False, default=8.0)  # maximum to ensure correct scaling
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0

    max_torque: float = 2.0
    maximum_max_torque: float = struct.field(pytree_node=False, default=2.0)  # maximum to ensure correct scaling


class PendulumCSDA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

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

        newthdot = (state.theta_dot + (-3 * params.gravity / (2 * params.length) * jnp.sin(state.theta + jnp.pi) +
                                       3.0 / (params.mass * params.length ** 2) * action) * params.dt)
        unnorm_newth = state.theta + newthdot * params.dt
        newth = self._angle_normalise(unnorm_newth)
        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)

        # delta_s = jnp.array((unnorm_newth, newthdot)) - self.get_obs(state)
        # TODO check why this is unnorm_newth and that from the original

        new_state = EnvState(theta=newth, theta_dot=newthdot, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    @staticmethod
    def _angle_normalise(x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        high = jnp.array([jnp.pi, 1])
        init_state = jrandom.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = EnvState(theta=init_state[0],
                         theta_dot=init_state[1],
                         time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        action_t = self.action_convert(input_action_t, params)
        costs = self._angle_normalise(state_tp1.theta) ** 2 + 0.1 * state_tp1.theta_dot ** 2 + 0.001 * (action_t ** 2)

        done = jnp.array(state_tp1.time >= params.max_steps_in_ep)  # TODO state_t or state_tp1

        return -costs, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return params.action_array[action.squeeze()] * params.max_torque

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array([state.theta, state.theta_dot])

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(theta=obs[0], theta_dot=obs[1], time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        def get_coords(theta, length):
            return -length * jnp.sin(theta), length * jnp.cos(theta)

        thetas = np.asarray(trajectory_state.theta)
        lengths = np.asarray(params.length)

        max_length = np.max(lengths)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        ax.set_xlim(-max_length * 1.2, max_length * 1.2)
        ax.set_xlabel("X")
        ax.set_ylim(-max_length * 1.2, max_length * 1.2)
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        ax.grid(True)
        x0, y0 = get_coords(thetas[0], lengths[0])
        line, = ax.plot([0, x0], [0, y0], lw=3, c='k')
        # time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        bob_radius = 0.08
        circle = ax.add_patch(plt.Circle((x0, y0), bob_radius, fc='r', zorder=3))

        def update(frame):
            x, y = get_coords(thetas[frame], lengths[frame])
            line.set_data([0, x], [0, y])
            circle.set_center((x, y))
            return line, circle

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=thetas.shape[0],
                                       interval=params.dt * 1000,  # Convert dt to milliseconds
                                       blit=True
                                       )
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "Pendulum-v0"

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(params.action_array))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        high = jnp.array([jnp.pi, params.maximum_max_speed])
        return spaces.Box(-high, high, (2,), dtype=jnp.float32)

    def reward_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-20, 0, (()), dtype=jnp.float32)


class PendulumCSCA(PendulumCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_torque, params.max_torque).squeeze()

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_torque, params.maximum_max_torque, shape=(1,))
