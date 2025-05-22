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
    x: jnp.ndarray
    y: jnp.ndarray
    time: int


class HenonMapDSDA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.start_point: float = 0.0
        self.random_start: bool = True
        self.random_start_range_lower: float = -1.5
        self.random_start_range_upper: float = 1.5

        self.discretisation = 100 + 1
        self.ref_vector: jnp.ndarray = jnp.linspace(0, 1, self.discretisation)

        self.reward_ball: float = 0.001
        self.init_a: float = 1.4
        self.init_b: float = 0.3

        self.fixed_point: jnp.ndarray = jnp.array((0.631354477,
                                                   0.189406343))

        self.action_array: jnp.ndarray = jnp.array(((0.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                                                    (1.0, 0.0), (1.0, 1.0), (1.0, -1.0),
                                                    (-1.0, 0.0), (-1.0, 1.0), (-1.0, -1.0)))
        self.max_control: float = 0.1

        self.horizon: int = 200

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        new_x = 1 - (self.init_a + action[0]) * jnp.square(state.x) + state.y

        new_y = (self.init_b + action[1]) * state.x

        new_state = EnvState(x=new_x, y=new_y, time=state.time+1)

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {})

    def projection(self, s):
        s = jnp.repeat(s, self.ref_vector.shape[0])
        inter = jnp.abs(self.ref_vector - s)
        return jnp.argmin(inter)

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=jnp.array(self.start_point), y=jnp.array(self.start_point), time=0)
        random_state = EnvState(x=jrandom.uniform(_key,
                                                  shape=(),
                                                  minval=self.random_start_range_lower,
                                                  maxval=self.random_start_range_upper),
                                y=jrandom.uniform(_key,
                                                  shape=(),
                                                  minval=self.random_start_range_lower,
                                                  maxval=self.random_start_range_upper),
                                time=0)

        state = jax.tree.map(lambda x, y: jax.lax.select(self.random_start, x, y), random_state, same_state)

        return self.get_obs(state), state

    def reward_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> chex.Array:
        reward = -jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y)) - self.fixed_point, 2)
        # the above can set more specific norm distances

        return reward

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action] * self.max_control

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((self.projection(state.x), self.projection(state.y)))

    def get_state(self, obs: chex.Array) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], time=-1)

    def is_done(self, state: EnvState) -> chex.Array:
        done_condition = jnp.logical_and(jnp.abs(state.x - self.fixed_point[0]) < self.reward_ball,
                                         jnp.abs(state.y - self.fixed_point[1]) < self.reward_ball,)
        return jax.lax.select(done_condition,  # TODO is there a better way to do this?
                              jnp.array(True),
                              jnp.array(False))

    @property
    def name(self) -> str:
        return "HenonMap-v0"

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.discretisation)


class HenonMapCSDA(HenonMapDSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x, state.y))

    def observation_space(self) -> spaces.Box:
        return spaces.Box(-1.5, 1.5, (2,), dtype=jnp.float64)


class HenonMapCSCA(HenonMapCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_control, self.max_control)

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, shape=(2,), dtype=jnp.float64)