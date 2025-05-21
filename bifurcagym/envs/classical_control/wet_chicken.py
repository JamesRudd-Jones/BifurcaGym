"""
Based off the following: https://github.com/LAVA-LAB/improved_spi/blob/main/wetChicken.py
"""

import numpy as np
from os import path
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from gymnax.environments import spaces
from flax import struct
from typing import Any, Dict, Optional, Tuple, Union
import chex
from jax import lax


ACTION_TRANSLATOR = {
    'Drift': np.zeros(2),
    'Neutral': np.array([-1, 0]),
    'Max': np.array([-2, 0]),
    'Left': np.array([0, -1]),
    'Right': np.array([0, 1])
}


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray
    time: int


class WetChickenDSDA(base_env.BaseEnvironment):
    # Implements the 2-dimensional Wet Chicken benchmark from 'Efficient Uncertainty Propagation for
    # Reinforcement Learning with Limited Data' by Alexander Hans and Steffen Udluft

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.length: float = 5.0
        self.width: float = 5.0
        self.max_turbulence: float = 3.5
        self.max_velocity: float = 3.
        self.action_high: float = 1.0
        self.hoirzon: int = 200


    def _velocity(self, state):
        return self.max_velocity * state.x / self.width

    def _turbulence(self, state):
        return self.max_turbulence - self._velocity(state)

    def step_env(self,
                 input_action: Union[int, float, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        action = self._action_convert(input_action)
        # if self.discrete:
        #     action_coordinates = list(ACTION_TRANSLATOR.values())[action]
        #     y_hat = self._state[1] + action_coordinates[1] + self._velocity() + self._turbulence() * np.random.uniform(
        #         -1, 1)
        #     x_hat = self._state[0] + action_coordinates[0]
        #     x_hat = round(x_hat)
        #     y_hat = round(y_hat)
        # else:
        key, _key = jrandom.split(key)
        y_hat = state.y + (action[1] - 1) + self._velocity(state) + self._turbulence(state) * jrandom.uniform(_key, minval=-1, maxval=1)
        x_hat = state.x + action[0]

        if y_hat >= self.length or x_hat < 0:
            x_new = 0
        elif x_hat >= self.width:
            x_new = self.width
        else:
            x_new = x_hat

        if y_hat >= self.length:
            y_new = 0
        elif y_hat < 0:
            y_new = 0
        else:
            y_new = y_hat
        # TODO reformulate these conditionals

        info = {"delta_obs": (x_hat, y_hat)}

        done = jnp.array(False)  # TODO add is terminal, not sure about this as pretty sure it auto resets

        reward = -(self.length - y_new)  # TODO pretty sure its the new state y_t

        state = EnvState(x=x_new, y=y_new, time=state.time+1)

        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state),
                jnp.array(reward),  # TODO check reward is the correct way around
                done,
                info)

    def generative_step_env(self,
                            action: Union[int, float, chex.Array],
                            obs: chex.Array,
                            key: chex.PRNGKey,
                            ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        state = EnvState(x=obs[0], y=obs[1], time=0)
        return self.step(key, state, action)

    def _action_convert(self, input_action):
        return self.action_array[input_action]

    def reward_function(self,
                    x_t: chex.Array,
                    x_tp1: chex.Array,
                    key: chex.PRNGKey,
                    ) -> chex.Array:
        return -(self.length - x_tp1[..., 1])

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        state = EnvState(x=jnp.zeros((1,)),
                        y=jnp.zeros((1,)),
                        time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, key=None) -> chex.Array:
        return jnp.array([state.x, state.y])

    @property
    def name(self) -> str:
        """Environment name."""
        return "WetChicken-v0"

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(ACTION_TRANSLATOR))

    def observation_space(self) -> spaces.Discrete:
        return spaces.Discrete(2)  # TODO what is this actually

class WetChickenCSDA(WetChickenDSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def observation_space(self) -> spaces.Box:
        low = jnp.array([0, 0], dtype=jnp.float64)
        high = jnp.array([self.length, self.width], dtype=jnp.float64)  # TODO check the ordering
        return spaces.Box(low, high, (2,), dtype=jnp.float64)

class WetChickenCSCA(WetChickenCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.action_high, self.action_high, shape=(2,))

    # def _get_overlap(self, interval_1, interval_2):
    #     return max(0, min(interval_1[1], interval_2[1]) - max(interval_1[0], interval_2[0]))
    #
    # def get_transition_function(self):
    #     if not self.discrete:
    #         raise AssertionError('You chose a continuous MDP, but requested the transition function.')
    #     nb_states = self.width * self.length
    #     nb_actions = len(ACTION_TRANSLATOR)
    #     P = np.zeros((nb_states, nb_actions, nb_states))
    #     for state in range(nb_states):
    #         x = int(state / self.length)
    #         y = state % self.width
    #         velocity = self.max_velocity * y / self.width
    #         turbulence = self.max_turbulence - velocity
    #
    #         for action_nb, action in enumerate(ACTION_TRANSLATOR.keys()):
    #             action_coordinates = ACTION_TRANSLATOR[action]
    #             target_interval = [x + action_coordinates[0] + velocity - turbulence,
    #                                x + action_coordinates[0] + velocity + turbulence]
    #             prob_mass_on_land = 1 / (2 * turbulence) * self._get_overlap([-self.max_turbulence - 2, -0.5],
    #                                                                          target_interval)  # -self.max_turbulence - 2 should be the lowest possible
    #             prob_mass_waterfall = 1 / (2 * turbulence) * self._get_overlap(
    #                 [self.length - 0.5, self.length + self.max_turbulence + self.max_velocity],
    #                 target_interval)  # self.length + self.max_turbulence + self.max_velocity should be the highest possible
    #             y_hat = y + action_coordinates[1]
    #             if y_hat < 0:
    #                 y_new = 0
    #             elif y_hat >= self.width:
    #                 y_new = self.width - 1
    #             else:
    #                 y_new = y_hat
    #             y_new = int(y_new)
    #             P[state, action_nb, 0] += prob_mass_waterfall
    #             P[state, action_nb, y_new] += prob_mass_on_land
    #             for x_hat in range(self.width):
    #                 x_hat_interval = [x_hat - 0.5, x_hat + 0.5]
    #                 prob_mass = 1 / (2 * turbulence) * self._get_overlap(
    #                     x_hat_interval, target_interval)
    #                 P[state, action_nb, x_hat * self.length + y_new] += prob_mass
    #     return P
