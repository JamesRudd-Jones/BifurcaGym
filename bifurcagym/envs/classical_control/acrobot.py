"""
Based off https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
and
https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/acrobot.py
"""


import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env, utils
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex


@struct.dataclass
class EnvState(base_env.EnvState):
    joint_angle_1: jnp.ndarray
    joint_angle_2: jnp.ndarray
    vel_1: jnp.ndarray
    vel_2: jnp.ndarray
    time: int


class AcrobotCSDA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.length_1: float = 1.0
        self.length_2: float = 1.0
        self.mass_1: float = 1.0
        self.mass_2: float = 1.0
        self.com_pos_1: float = 0.5
        self.com_pos_2: float = 0.5
        self.moi: float = 1.0
        self.max_vel_1: float = 4 * jnp.pi
        self.max_vel_2: float = 9 * jnp.pi

        self.action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))
        self.max_torque: float = 1.0
        self.torque_noise_max: float = 0.0

        self.max_steps_in_episode: int = 500

        self.dt: float = 0.2

    def step_env(self,
                 input_action: Union[int, float, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        torque = self.action_convert(input_action)

        # Add noise to force action
        torque = torque + jrandom.uniform(key,
                                          shape=(),
                                          minval=-self.torque_noise_max,
                                          maxval=self.torque_noise_max)

        # Augment state with force action so it can be passed to ds/dt
        s_augmented = jnp.array([state.joint_angle_1,
                                 state.joint_angle_2,
                                 state.vel_1,
                                 state.vel_2,
                                 torque,
                                 ])

        ns = utils.runge_kutta_4(s_augmented, self._dsdt, self.dt)
        joint_angle_1 = self._wrap(ns[0], -jnp.pi, jnp.pi)
        joint_angle_2 = self._wrap(ns[1], -jnp.pi, jnp.pi)
        vel_1 = jnp.clip(ns[2], -self.max_vel_1, self.max_vel_1)
        vel_2 = jnp.clip(ns[3], -self.max_vel_2, self.max_vel_2)

        new_state = EnvState(joint_angle_1=joint_angle_1,
                             joint_angle_2=joint_angle_2,
                             vel_1=vel_1,
                             vel_2=vel_2,
                             time=jnp.int32(state.time + 1),
                             )

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {"discount": self.discount(new_state)},
                )

    def _dsdt(self, s_augmented: chex.Array, _: jnp.float_) -> chex.Array:
        m1, m2 = self.mass_1, self.mass_2
        l1 = self.length_1
        lc1, lc2 = self.com_pos_1, self.com_pos_2
        i1, i2 = self.moi, self.moi
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1, theta2, dtheta1, dtheta2 = s
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * jnp.cos(theta2)) + i1 + i2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * jnp.cos(theta2)) + i2
        phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
        phi1 = (-m2 * l1 * lc2 * dtheta2 ** 2 * jnp.sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
                + phi2
                )
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * jnp.sin(theta2) - phi2
                   ) / (m2 * lc2 ** 2 + i2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])

    def _wrap(self, x: jnp.float_, m: jnp.float_, big_m: jnp.float_) -> chex.Array:
        diff = big_m - m
        go_up = x < m
        go_down = x >= big_m

        how_often = go_up * jnp.ceil((m - x) / diff) + go_down * jnp.floor( (x - big_m) / diff + 1)
        x_out = x - how_often * diff * go_down + how_often * diff * go_up
        return x_out

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        init_state = jrandom.uniform(key, shape=(4,), minval=-0.1, maxval=0.1)
        state = EnvState(joint_angle_1=init_state[0],
                         joint_angle_2=init_state[1],
                         vel_1=init_state[2],
                         vel_2=init_state[3],
                         time=0)

        return self.get_obs(state), state

    def reward_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        )-> chex.Array:
        done_angle = -jnp.cos(state_tp1.joint_angle_1) - jnp.cos(state_tp1.joint_angle_2 + state_tp1.joint_angle_1) > 1.0
        reward = -1.0 * (1 - done_angle)

        return reward

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action] * self.max_torque

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((jnp.cos(state.joint_angle_1),
                          jnp.sin(state.joint_angle_1),
                          jnp.cos(state.joint_angle_2),
                          jnp.sin(state.joint_angle_2),
                          state.vel_1,
                          state.vel_2))

    def get_state(self, obs: chex.Array) -> EnvState:
        return EnvState(joint_angle_1=jnp.arctan2(obs[1], obs[0]),
                        joint_angle_2=jnp.arctan2(obs[3], obs[2]),
                        vel_1=obs[4],
                        vel_2=obs[5],
                        time=-1)

    def is_done(self, state: EnvState) -> chex.Array:
        # Check termination and construct updated state
        # done_angle = (
        #         -jnp.cos(state.joint_angle1)
        #         - jnp.cos(state.joint_angle2 + state.joint_angle1)
        #         > 1.0
        # )
        # # Check number of steps in episode termination condition
        # done_steps = state.time >= params.max_steps_in_episode
        # done = jnp.logical_or(done_angle, done_steps)
        # return done
        return jnp.array(False)

    def render_traj(self, trajectory_state: EnvState):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        def _get_coords(theta, length):
            return -length * jnp.sin(theta), length * jnp.cos(theta)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        screen_lim = self.length_1 + self.length_2
        ax.set_xlim(-screen_lim * 3, screen_lim * 3)
        ax.set_xlabel("X")
        ax.set_ylim(-screen_lim* 1.2, screen_lim * 1.2)
        # ax.set_ylabel("Y")
        ax.set_aspect('equal')
        ax.grid(True)

        line, = ax.plot([], [], 'k-', lw=3, zorder=3)
        dot, = ax.plot([], [], color="r", marker="o", markersize=8, zorder=4, label='Current State')

        def update(frame):
            x_1, y_1 = _get_coords(trajectory_state.joint_angle_1[frame], self.length_1)
            x_2, y_2 = _get_coords(trajectory_state.joint_angle_2[frame], self.length_2)

            line.set_data([0.0, -x_1, -x_2-x_1], [0.0, -y_1, -y_2-y_1])
            dot.set_data([-x_1-x_2], [-y_1-y_2])

            return line, dot

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
        return "Acrobot-v0"

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self) -> spaces.Box:
        high = jnp.array([1.0,
                                  1.0,
                                  1.0,
                                  1.0,
                                  self.max_vel_1,
                                  self.max_vel_2], dtype=jnp.float32)
        return spaces.Box(-high, high, (6,), jnp.float32)


class AcrobotCSCA(AcrobotCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_torque, self.max_torque).squeeze()

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_torque, self.max_torque, shape=(1,))


class PendubotCSDA(AcrobotCSDA):
    """
    The shoulder is actuated instead of the elbow.

    https://underactuated.mit.edu/acrobot.html
    https://link.springer.com/chapter/10.1007/BFb0015081
    """
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.max_torque: float = 1.0

    def _dsdt(self, s_augmented: chex.Array, _: jnp.float_) -> chex.Array:
        m1, m2 = self.mass_1, self.mass_2
        l1 = self.length_1
        lc1, lc2 = self.com_pos_1, self.com_pos_2
        i1, i2 = self.moi, self.moi
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1, theta2, dtheta1, dtheta2 = s
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * jnp.cos(theta2)) + i1 + i2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * jnp.cos(theta2)) + i2
        phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
        phi1 = (-m2 * l1 * lc2 * dtheta2 ** 2 * jnp.sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
                + phi2
                )
        ddtheta2 = (d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * jnp.sin(theta2) - phi2
                   ) / (m2 * lc2 ** 2 + i2 - d2 ** 2 / d1)
        ddtheta1 = -(a + d2 * ddtheta2 + phi1) / d1

        return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])

    @property
    def name(self) -> str:
        return "Pendubot-v0"


class PendubotCSCA(PendubotCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_torque, self.max_torque).squeeze()

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_torque, self.max_torque, shape=(1,))