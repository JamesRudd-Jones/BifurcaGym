"""
Based off: https://github.com/fusion-ml/trajectory-information-rl/blob/main/barl/envs/pilco_cartpole.py
"""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
import jax.numpy as jnp
from bifurcagym.envs import base_env
from gymnax.environments import spaces


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int


class PilcoCartPoleCSDA(base_env.BaseEnvironment):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.periodic_dim = jnp.array((0, 0, 1, 0))  # TODO is this the best way?

        self.gravity: float = 9.82
        self.masscart: float = 0.5  # 1.0
        self.masspole: float = 0.5  # 0.1
        self.total_mass: float = self.masscart + self.masspole 
        self.length: float = 0.6  # 0.5
        self.mass_pole_length: float = self.masspole * self.length  
        self.force_mag: float = 10.0
        self.dt: float = 0.1  
        self.b: float = 0.1  # friction coefficient
        self.theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
        self.x_threshold: float = 2.4
        self.horizon: int = 25

        self.action_array: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))

    def step_env(self,
                 input_action: Union[int, float, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        action = self._action_convert(input_action)

        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        xdot_update = (-2 * self.mass_pole_length * (state.theta_dot ** 2) * sintheta + 3 * self.masspole * self.gravity
                        * sintheta * costheta + 4 * action - 4 * self.b * state.x_dot) / (4 * self.total_mass - 3
                                                                                            * self.masspole * costheta ** 2)

        thetadot_update = (-3 * self.mass_pole_length * (state.theta_dot ** 2) * sintheta * costheta + 6 * self.total_mass
                           * self.gravity * sintheta + 6 * (action - self.b * state.x_dot) * costheta) / (4 * self.length * self.total_mass - 3 * self.mass_pole_length * costheta ** 2)

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

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                delta_s,
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {"discount": self.discount(new_state)})

    def generative_step_env(self,
                            action: Union[int, float, chex.Array],
                            obs: chex.Array,
                            key: chex.PRNGKey,
                            ) -> Tuple[chex.Array, chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        state = EnvState(x=obs[0], x_dot=obs[1], theta=self._angle_normalise(obs[2]), theta_dot=obs[3], time=0)
        return self.step(action, state, key)

    def _action_convert(self, input_action):
        return self.action_array[input_action] * self.force_mag

    @staticmethod
    def _angle_normalise(x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def reward_function(self,
                    input_action_t: Union[int, float, chex.Array],
                    state_t: EnvState,
                    state_tp1: EnvState,
                    key: chex.PRNGKey,
                    ) -> chex.Array:
        goal = jnp.array([0.0, self.length])
        pole_x = self.length * jnp.sin(state_tp1.theta)
        pole_y = self.length * jnp.cos(state_tp1.theta)
        position = jnp.array([state_tp1.x + pole_x, pole_y])
        squared_distance = jnp.sum((position - goal) ** 2)
        squared_sigma = 0.25 ** 2
        costs = 1 - jnp.exp(-0.5 * squared_distance / squared_sigma)

        return -costs

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        loc = jnp.array([0.0, 0.0, jnp.pi, 0.0])
        scale = jnp.array([0.02, 0.02, 0.02, 0.02])
        init_state = jax.random.normal(key, shape=(4,)) * scale + loc
        state = EnvState(x=init_state[0],
                         x_dot=init_state[1],
                         theta=self._angle_normalise(init_state[2]),
                         theta_dot=init_state[3],
                         time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, key=None) -> chex.Array:
        # TODO if self.use_trig then it is the below
        # return jnp.array([state.x, state.x_dot, jnp.sin(state.theta), jnp.cos(state.theta), state.theta_dot])
        # Otherwise
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    def is_done(self, state: EnvState) -> chex.Array:
        return jnp.array(False)

    @property
    def name(self) -> str:
        return "PilcoCartPole-v0"

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self) -> spaces.Box:
        high = jnp.array([10.0, 10.0, 3.14159, 25.0])
        return spaces.Box(-high, high, (4,), dtype=jnp.float32)


class PilcoCartPoleCSCA(PilcoCartPoleCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def _action_convert(self, input_action):
        return jnp.clip(input_action, -1, 1)[0] * self.force_mag

    def action_space(self) -> spaces.Box:
        return spaces.Box(-1, 1, shape=(1,))