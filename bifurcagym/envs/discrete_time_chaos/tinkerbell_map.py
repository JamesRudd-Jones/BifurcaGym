import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex
from bifurcagym.envs import utils


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray
    time: int


class TinkerbellMapCSCA(base_env.BaseEnvironment):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.a: float = 0.9
        self.b: float = -0.6013
        self.c: float = 2.0
        self.d: float = 0.5

        self.max_control: float = 0.05  # perturbation to a

        self.reward_ball: float = 1e-3
        self.max_steps_in_episode: int = 2000

        self.start_point = jnp.array([0.1, 0.1], dtype=jnp.float64)
        self.random_start: bool = False  # TODO turn it into an env kwargs
        self.random_start_low = jnp.array([-1.0, -1.0], dtype=jnp.float64)
        self.random_start_high = jnp.array([1.0, 1.0], dtype=jnp.float64)

        self.horizon: int = 200

        # Precompute a fixed point (solve map(s) - s = 0)
        self.fixed_point = self._compute_fixed_point()

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        new_xy = self._map(jnp.array((state.x, state.y)), action)
        new_state = EnvState(x=new_xy[0], y=new_xy[1], time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        obs_new = self.get_obs(new_state)
        obs_old = self.get_obs(state)

        return (jax.lax.stop_gradient(obs_new),
                jax.lax.stop_gradient(obs_new - obs_old),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def _map(self, xy: chex.Array, u: chex.Array) -> chex.Array:
        x = xy[0]; y = xy[1]
        xn1 = x * x - y * y + (self.a + u[0]) * x + self.b * y
        yn1 = 2.0 * x * y + (self.c + u[1]) * x + self.d * y

        return jnp.array((xn1, yn1))

    def _compute_fixed_point(self) -> chex.Array:
        # Fixed points satisfy map(s, u=0) - s = 0
        def F(xy):
            return self._map(xy, jnp.zeros(2)) - xy

        init_seeds = jnp.array(((0.0, 2.0),
                                (0.2, 0.2),
                                (-0.2, 0.2),
                                (0.5, -0.5)), dtype=jnp.float64)
        # TODO should the above be randomised idk with more options?

        sols = jax.vmap(utils.newton_solve_2d, in_axes=(None, 0))(F, init_seeds)
        residuals = jax.vmap(lambda x: jnp.linalg.norm(F(x)))(sols)
        best = int(jnp.argmin(residuals))

        # sols = [utils.newton_solve_2d(F, s0) for s0 in seeds]
        # residuals = [jnp.linalg.norm(F(sol)) for sol in sols]
        # best = int(jnp.argmin(jnp.array(residuals)))

        return sols[best]

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=self.start_point[0], y=self.start_point[1], time=0)
        rand = jrandom.uniform(_key, shape=(2,), minval=self.random_start_low, maxval=self.random_start_high)
        random_state = EnvState(x=rand[0], y=rand[1], time=0)

        state = jax.tree.map(lambda r, s: jax.lax.select(self.random_start, r, s), random_state, same_state)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        err = jnp.array((state_tp1.x, state_tp1.y)) - self.fixed_point
        reward = -jnp.sum(err * err)

        state_done = jnp.linalg.norm(err) < self.reward_ball
        time_done = state_tp1.time >= self.max_steps_in_episode

        diverged = jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y))) > 1e3

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
        return "TinkerbellMap-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, shape=(2,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        # TinkerbellMap is unbounded in principle so giving wide bounds  # TODO unsure how to fix this for normalisation
        return spaces.Box(-1e6, 1e6, shape=(2,), dtype=jnp.float64)


class TinkerbellMapCSDA(TinkerbellMapCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: jnp.ndarray = jnp.array(((0.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                                                    (1.0, 0.0), (1.0, 1.0), (1.0, -1.0),
                                                    (-1.0, 0.0), (-1.0, 1.0), (-1.0, -1.0)))

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action.squeeze()] * self.max_control

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))