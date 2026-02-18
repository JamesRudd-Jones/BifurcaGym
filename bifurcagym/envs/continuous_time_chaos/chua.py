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
    x: jnp.ndarray  # shape (3,)
    time: int


class ChuaCSCA(base_env.BaseEnvironment):
    """
    One common dimensionless Chua form:
      x' = alpha (y - x - f(x))
      y' = x - y + z
      z' = -beta y

    with piecewise-linear nonlinearity:
      f(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)

    Control u perturbs alpha: alpha_eff = alpha + u (clipped).
    """

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # typical double-scroll-ish set (dimensionless)
        self.alpha: float = 15.6
        self.beta: float = 28.0
        self.m0: float = -1.143
        self.m1: float = -0.714

        self.dt: float = 0.01
        self.substeps: int = 5

        self.max_control: float = 2.0  # 5.0  # alpha perturbation bound
        self.num_actions: int = 2

        self.max_steps_in_episode: int = int(200 // self.dt)
        self.reward_ball: float = 1e-2

        self.start_point = jnp.array([0.1, 0.0, 0.0], dtype=jnp.float64)
        self.random_start: bool = True  # TODO turn it into an env kwargs
        self.random_start_low = jnp.array([-2.0, -2.0, -2.0], dtype=jnp.float64)
        self.random_start_high = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float64)

        self.fixed_point = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)

        self.horizon: int = 200

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        u = self.action_convert(input_action)

        new_x = utils.integrate_ode(self._f, state.x, u, self.dt, self.substeps)
        new_state = EnvState(x=new_x, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        obs_new = self.get_obs(new_state)
        obs_old = self.get_obs(state)

        return (jax.lax.stop_gradient(obs_new),
                jax.lax.stop_gradient(obs_new - obs_old),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def _f_nl(self, x: chex.Array) -> chex.Array:
        """Chua piecewise-linear diode characteristic f(x)."""
        m0, m1 = self.m0, self.m1
        return m1 * x + 0.5 * (m0 - m1) * (jnp.abs(x + 1.0) - jnp.abs(x - 1.0))

    def _f(self, x: chex.Array, u: chex.Array) -> chex.Array:
        X, Y, Z = x[0], x[1], x[2]
        fx = self._f_nl(X)
        dx = (self.alpha + u[0]) * (Y - X - fx)
        dy = X - Y + Z
        dz = -(self.beta + u[1]) * Y
        return jnp.array([dx, dy, dz], dtype=jnp.float64)

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=self.start_point, time=0)
        rand_x = jrandom.uniform(_key, shape=(3,), minval=self.random_start_low, maxval=self.random_start_high)
        random_state = EnvState(x=rand_x, time=0)
        state = jax.tree.map(lambda r, s: jax.lax.select(self.random_start, r, s), random_state, same_state)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        err = state_tp1.x - self.fixed_point
        reward = -jnp.linalg.norm(err)  # -jnp.sum(err * err)

        state_done = jnp.linalg.norm(err) < self.reward_ball
        time_done = state_tp1.time >= self.max_steps_in_episode
        boundary_done = jnp.linalg.norm(state_tp1.x) > 1e3
        done = jnp.logical_or(jnp.logical_or(state_done, boundary_done), time_done)

        return reward, done

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_control, self.max_control).squeeze()

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.asarray(state.x)

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        return EnvState(x=jnp.asarray(obs), time=-1)

    @property
    def name(self) -> str:
        return "Chua-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, shape=(self.num_actions,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        # Rossler is unbounded in principle so giving wide bounds  # TODO unsure how to fix this for normalisation
        return spaces.Box(-1e6, 1e6, shape=(3,), dtype=jnp.float64)


class ChuaCSDA(ChuaCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))

        idx = jnp.arange(self.action_array.shape[0] ** self.num_actions)
        powers = self.action_array.shape[0] ** jnp.arange(self.num_actions)
        digits = (idx[:, None] // powers[None, :]) % self.action_array.shape[0]
        self.action_perms: jnp.ndarray = self.action_array[digits]
        # TODO should I add the following to utils to standardise it?

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_perms[action.squeeze()] * self.max_control

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_perms))