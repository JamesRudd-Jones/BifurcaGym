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


@struct.dataclass
class EnvParams:
    action_array: chex.Array = struct.field(False, default=jnp.array((0.0, 1.0, -1.0)))
    discretisation: int = struct.field(False, default=100 + 1)
    horizon: int = struct.field(False, default=200)

    start_point: float = struct.field(False, default=0.1)
    random_start: bool = struct.field(False, default=True)
    reward_ball: float = struct.field(False, default=0.001)

    init_mu: float = 2

    random_start_range_lower: float = 0.0
    random_start_range_upper: float = 1.0

    max_control: float = 0.1
    maximum_max_control: float = struct.field(False, default=0.1)  # maximum to ensure correct scaling

    @property
    def fixed_point(self) -> float:
        return self.init_mu / (self.init_mu + 1)  # TODO is this true?

    @property
    def ref_vector(self) -> chex.Array:
        return jnp.linspace(0, 1, self.discretisation)


class TentMapCSCA(base_env.BaseEnvironment):

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

        new_x = (action + params.init_mu) * jnp.min(jnp.array((state.x, 1 - state.x)))

        new_state = EnvState(x=new_x, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=jnp.array(params.start_point), time=0)
        random_state = EnvState(x=jrandom.uniform(_key,
                                                  shape=(),
                                                  minval=params.random_start_range_lower,
                                                  maxval=params.random_start_range_upper),
                                time=0)

        state = jax.tree.map(lambda x, y: jax.lax.select(params.random_start, x, y), random_state, same_state)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        reward = -jnp.abs(state_tp1.x - params.fixed_point) ** 2
        # The above can set more specific norm distances

        # TODO state_t or state_tp1
        done =  jax.lax.select(jnp.abs(state_tp1.x - params.fixed_point) < params.reward_ball,
                              jnp.array(True),
                              jnp.array(False))

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control).squeeze()

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x,))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(x=obs[0], time=0)

    @property
    def name(self) -> str:
        return "TentcMap-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(1,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (1,), dtype=jnp.float64)


class TentMapCSDA(TentMapCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return params.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(params.action_array))


class TentMapDSDA(TentMapCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def generative_step(self,
                        action: chex.Numeric,
                        gen_obs: chex.Array,
                        params: EnvParams,
                        key: chex.PRNGKey,
                        ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        raise ValueError(f"No Generative Step for {self.name} Discrete State.")

    def _projection(self, s):
        inter = jnp.abs(self.ref_vector - s)
        return jnp.argmin(inter)

    def get_obs(self, state: EnvState, key=None) -> chex.Array:
        return jnp.array((self._projection(state.x),))
        # TODO sort out the above

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(params.discretisation)