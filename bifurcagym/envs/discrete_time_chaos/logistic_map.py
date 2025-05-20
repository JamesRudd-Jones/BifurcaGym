import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from gymnax.environments import spaces
from flax import struct
from typing import Any, Dict, Optional, Tuple, Union
import chex
import jax


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    time: int
    

class LogisticMapDSDA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # init_r: float = 2.5
        # init_r: float = 3.1
        self.init_r: float = 3.8
        # init_r: float = 4.0

        # fixed_point: float = 0.6  # for r = 2.5; period 1
        # fixed_point: float = 0.55801  # for r = 3.1; period 2
        # fixed_point: float = 0.76457  # for r = 3.1; period 2
        # fixed_point: float = 0.67742  # for r = 3.1; period 1
        # self.fixed_point: float = 0.737  # for r = 3.8; chaotic

        self.fixed_point: float = (self.init_r - 1) / self.init_r
        # TODO a calc for period fixed point but be good to generalise to more

        self.reward_ball: float = 0.001
        self.max_control: float = 0.1
        self.horizon: int = 200
        self.action_array: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))
        self.discretisation = 100 + 1
        self.ref_vector: jnp.ndarray = jnp.linspace(0, 1, self.discretisation)

        self.start_point: float = 0.1
        self.random_start: bool = True
        self.random_start_range_lower: float = 0.0
        self.random_start_range_upper: float = 1.0

    def step_env(self,
                 input_action: Union[int, float, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        action = self._action_convert(input_action).squeeze()

        new_x = (action + self.init_r) * state.x * (1 - state.x)

        new_state = EnvState(x=new_x, time=state.time+1)

        reward = self.reward_func(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {})

    def generative_step_env(self,
                            action: Union[int, float, chex.Array],
                            obs: chex.Array,
                            key: chex.PRNGKey,
                            ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        state = EnvState(x=obs, time=0)
        return self.step(action, state, key)

    def _action_convert(self, input_action):
        return self.action_array[input_action] * self.max_control

    def reward_func(self,
                    input_action_t: Union[int, float, chex.Array],
                    state_t: EnvState,
                    state_tp1: EnvState,
                    key: chex.PRNGKey,
                    ) -> chex.Array:
        """
        As per the paper titled: Optimal chaos control through reinforcement learning
        """
        reward = -jnp.abs(state_tp1.x - self.fixed_point) ** 2
        # The above can set more specific norm distances
        return reward

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=jnp.array(self.start_point), time=0)
        random_state = EnvState(x=jrandom.uniform(_key,
                                                  shape=(),
                                                  minval=self.random_start_range_lower,
                                                  maxval=self.random_start_range_upper),
                                time=0)

        state = jax.tree.map(lambda x, y: jax.lax.select(self.random_start, x, y), random_state, same_state)

        return self.get_obs(state), state

    def _projection(self, s):
        s = jnp.repeat(s, self.ref_vector.shape[0])
        inter = jnp.abs(self.ref_vector - s)
        return jnp.argmin(inter, keepdims=True)

    def get_obs(self, state: EnvState, key=None) -> chex.Array:
        return self._projection(state.x)

    def is_done(self, state: EnvState) -> jnp.ndarray:
        return jax.lax.select(jnp.abs(state.x - self.fixed_point) < self.reward_ball,
                              jnp.array(True),
                              jnp.array(False))

    @property
    def name(self) -> str:
        return "LogisticMap-v0"

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.discretisation)


class LogisticMapCSDA(LogisticMapDSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def get_obs(self, state: EnvState, key=None) -> chex.Array:
        return state.x

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0.0, 1.0, (1,), dtype=jnp.float32)


class LogisticMapCSCA(LogisticMapCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def _action_convert(self, input_action):
        return jnp.clip(input_action, -self.max_control, self.max_control)

    def action_space(self) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-self.max_control, self.max_control, shape=(1,))