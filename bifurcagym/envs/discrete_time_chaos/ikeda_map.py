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


class IkedaMapCSCA(base_env.BaseEnvironment):
    """
    Ikeda map:
      x_{n+1} = 1 + u (x cos(tau) - y sin(tau))
      y_{n+1} =     u (x sin(tau) + y cos(tau))
      tau = k - p / (1 + x^2 + y^2)

    Control action perturbs parameter u: u_eff = u + a (clipped), where u is the Ikeda parameter.
    """

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.u_param: float = 0.9#18
        self.k: float = 0.4
        self.p: float = 6.0

        self.fixed_point: jnp.ndarray = jnp.array((0.532754622941,
                                                   0.246896772711))

        self.max_control: float = 0.1  # perturbation to u_param
        self.u_min: float = 0.0
        self.u_max: float = 0.99  # keep in typical stable range

        self.reward_ball: float = 1e-3

        self.start_point = jnp.array([0.1, 0.1], dtype=jnp.float64)
        self.random_start: bool = False  # TODO turn it into an env kwargs
        self.random_start_low = jnp.array([-2.0, -2.0], dtype=jnp.float64)
        self.random_start_high = jnp.array([2.0, 2.0], dtype=jnp.float64)

        self.max_steps_in_episode: int = 2000

        self.horizon: int = 200

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        u_eff = jnp.clip(self.u_param + action, self.u_min, self.u_max)

        r2 = 1.0 + state.x * state.x + state.y * state.y
        tau = self.k - self.p / r2

        ct = jnp.cos(tau)
        st = jnp.sin(tau)

        new_x = 1.0 + u_eff * (state.x * ct - state.y * st)
        new_y = u_eff * (state.x * st + state.y * ct)

        new_state = EnvState(x=new_x, y=new_y, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        obs_new = self.get_obs(new_state)
        obs_old = self.get_obs(state)

        return (jax.lax.stop_gradient(obs_new),
                jax.lax.stop_gradient(obs_new - obs_old),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=self.start_point[0], y=self.start_point[1], time=0)
        rand_start = jrandom.uniform(_key, shape=(2,), minval=self.random_start_low, maxval=self.random_start_high)
        random_state = EnvState(x=rand_start[0], y=rand_start[1], time=0)

        state = jax.tree.map(lambda r, s: jax.lax.select(self.random_start, r, s), random_state, same_state)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        err = jnp.array((state_tp1.x, state_tp1.y)) - self.fixed_point
        reward = -jnp.linalg.norm(err)  # -jnp.sum(err * err)

        state_done = jnp.linalg.norm(err) < self.reward_ball
        time_done = state_tp1.time >= self.max_steps_in_episode

        diverged = jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y))) > 1e4

        done = jnp.logical_or(jnp.logical_or(state_done, time_done), diverged)

        return reward, done

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_control, self.max_control).squeeze()

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x, state.y))

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], time=-1)

    @property
    def name(self) -> str:
        return "IkedaMap-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, shape=(1,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        # IkedaMap is unbounded in principle so giving wide bounds  # TODO unsure how to fix this for normalisation
        return spaces.Box(-1e6, 1e6, shape=(2,), dtype=jnp.float64)


class IkedaMapCSDA(IkedaMapCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: jnp.ndarray = jnp.array((0.0, 1.0, 0.5, -1.0, -0.5))

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action.squeeze()] * self.max_control

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
