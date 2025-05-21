import numpy as np
from os import path
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env, utils
from gymnax.environments import spaces
from flax import struct
from typing import Any, Dict, Optional, Tuple, Union
import chex
import jax
import matplotlib.pyplot as plt


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

        self.dt: float = 0.2
        self.link_length_1: float = 1.0
        self.link_length_2: float = 1.0
        self.link_mass_1: float = 1.0
        self.link_mass_2: float = 1.0
        self.link_com_pos_1: float = 0.5
        self.link_com_pos_2: float = 0.5
        self.link_moi: float = 1.0
        self.max_vel_1: float = 4 * jnp.pi
        self.max_vel_2: float = 9 * jnp.pi
        self.torque_noise_max: float = 0.0
        self.max_steps_in_episode: int = 500

        self.action_array: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))
        self.max_torque: float = 1.0

    def step_env(self,
                 input_action: Union[int, float, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        torque = self._action_convert(input_action)

        # Add noise to force action - always sample - conditionals in JAX
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
                jnp.array(None),  # TODO add delta obs
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {"discount": self.discount(new_state)},
                )

    def generative_step_env(self,
                            action: Union[int, float, chex.Array],
                            obs: chex.Array,
                            key: chex.PRNGKey,
                            ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        state = EnvState(joint_angle_1=jnp.arctan2(obs[1], obs[0]),
                         joint_angle_2=jnp.arctan2(obs[3], obs[2]),
                         vel_1=obs[4],
                         vel_2=obs[5],
                         time=0)
        return self.step(action, state, key)

    def _dsdt(self, s_augmented: chex.Array, _: float) -> chex.Array:
        """Compute time derivative of the state change - Use for ODE int."""
        m1, m2 = self.link_mass_1, self.link_mass_2
        l1 = self.link_length_1
        lc1, lc2 = self.link_com_pos_1, self.link_com_pos_2
        i1, i2 = self.link_moi, self.link_moi
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1, theta2, dtheta1, dtheta2 = s
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * jnp.cos(theta2)) + i1 + i2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * jnp.cos(theta2)) + i2
        phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
        phi1 = (
                -m2 * l1 * lc2 * dtheta2 ** 2 * jnp.sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
                + phi2
        )
        ddtheta2 = (
                           a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * jnp.sin(theta2) - phi2
                   ) / (m2 * lc2 ** 2 + i2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])

    def _wrap(self, x: float, m: float, big_m: float) -> chex.Array:
        """For example, m = -180, M = 180 (degrees), x = 360 --> returns 0."""
        diff = big_m - m
        go_up = x < m  # Wrap if x is outside the left bound
        go_down = x >= big_m  # Wrap if x is outside OR on the right bound

        how_often = go_up * jnp.ceil(
            (m - x) / diff
        ) + go_down * jnp.floor(  # if m - x is an integer, keep it
            (x - big_m) / diff + 1
        )  # if x - M is an integer, round up
        x_out = x - how_often * diff * go_down + how_often * diff * go_up
        return x_out

    def _action_convert(self, input_action):
        return self.action_array[input_action] * self.max_torque

    def reward_function(self,
                    input_action_t: Union[int, float, chex.Array],
                    state_t: EnvState,
                    state_tp1: EnvState,
                    key: chex.PRNGKey,
                    )-> chex.Array:
        done_angle = -jnp.cos(state_tp1.joint_angle_1) - jnp.cos(state_tp1.joint_angle_2 + state_tp1.joint_angle_1) > 1.0
        reward = -1.0 * (1 - done_angle)

        return reward

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        init_state = jax.random.uniform(key, shape=(4,), minval=-0.1, maxval=0.1)
        state = EnvState(joint_angle_1=init_state[0],
                         joint_angle_2=init_state[1],
                         vel_1=init_state[2],
                         vel_2=init_state[3],
                         time=0)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState, key=None) -> chex.Array:
        return jnp.array((jnp.cos(state.joint_angle_1),
                          jnp.sin(state.joint_angle_1),
                          jnp.cos(state.joint_angle_2),
                          jnp.sin(state.joint_angle_2),
                          state.vel_1,
                          state.vel_2))

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

    def _action_convert(self, input_action):
        jnp.clip(input_action, -self.max_torque, self.max_torque)

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_torque, self.max_torque, shape=(1,))