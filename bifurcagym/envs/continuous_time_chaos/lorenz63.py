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


@struct.dataclass
class EnvParams:
    action_array: chex.Array = struct.field(False, default=jnp.array((0.0, 1.0, -1.0)))
    dt: float = struct.field(False, default=0.01)
    horizon: int = struct.field(False, default=200)
    substeps: int = struct.field(False, default=5)

    num_actions: int = struct.field(False, default=3)
    start_point: chex.Array = struct.field(False, default=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64))
    random_start: bool = struct.field(False, default=False)
    fixed_point: chex.Array = struct.field(False, default=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64))
    reward_ball: float = struct.field(False, default=1e-2)

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    max_control: chex.Array = jnp.array((2.0, 5.0, 0.5))  # 2.0  # 5.0  # rho perturbation bound
    maximum_max_control: chex.Array = struct.field(False, default=2.0)  # maximum to ensure correct scaling

    random_start_low: chex.Array = jnp.array([-10.0, -10.0, 0.0], dtype=jnp.float64)
    random_start_high: chex.Array = jnp.array([10.0, 10.0, 30.0], dtype=jnp.float64)

    def action_perms(self) -> chex.Array:
        idx = jnp.arange(self.action_array.shape[0] ** self.num_actions)
        powers = self.action_array.shape[0] ** jnp.arange(self.num_actions)
        digits = (idx[:, None] // powers[None, :]) % self.action_array.shape[0]
        return self.action_array[digits]
        # TODO this does not scale very nicely
        # TODO should I add the following to utils to standardise it? if it is okay to use

    @property
    def max_steps_in_ep(self) -> int:
        return int(200 // self.dt)


class Lorenz63CSCA(base_env.BaseEnvironment):
    """
    Lorenz system:
      x' = sigma (y - x)
      y' = x (rho - z) - y
      z' = x y - beta z

    Control u perturbs rho: rho_eff = rho + u (clipped).
    """

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
        u = self.action_convert(input_action, params)

        new_x = utils.integrate_ode(self._f, state.x, u, params.dt, params.substeps, params)
        new_state = EnvState(x=new_x, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        obs_new = self.get_obs(new_state)
        obs_old = self.get_obs(state)

        return (jax.lax.stop_gradient(obs_new),
                jax.lax.stop_gradient(obs_new - obs_old),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def _f(self, x: chex.Array, u: chex.Array, params: EnvParams) -> chex.Array:
        X, Y, Z = x[0], x[1], x[2]
        dx = (params.sigma + u[0]) * (Y - X)
        dy = X * ((params.rho + u[1]) - Z) - Y
        dz = X * Y - (params.beta + u[2]) * Z
        return jnp.array([dx, dy, dz], dtype=jnp.float64)

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=params.start_point, time=0)
        rand_x = jrandom.uniform(_key, shape=(3,), minval=params.random_start_low, maxval=params.random_start_high, dtype=jnp.float64)
        random_state = EnvState(x=rand_x, time=0)
        state = jax.tree.map(lambda r, s: jax.lax.select(params.random_start, r, s), random_state, same_state)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        err = state_tp1.x - params.fixed_point
        reward = -jnp.linalg.norm(err)  # -jnp.sum(err * err)

        state_done = jnp.linalg.norm(err) < params.reward_ball
        time_done = state_tp1.time >= params.max_steps_in_ep
        boundary_done = jnp.linalg.norm(state_tp1.x) > 1e3
        done = jnp.logical_or(jnp.logical_or(state_done, boundary_done), time_done)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control).squeeze()

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.asarray(state.x)

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(x=jnp.asarray(obs), time=-1)

    @property
    def name(self) -> str:
        return "Lorenz63-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(params.num_actions,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        # Lorenz is unbounded in principle so giving wide bounds  # TODO unsure how to fix this for normalisation
        return spaces.Box(-1e6, 1e6, shape=(3,), dtype=jnp.float64)


class Lorenz63CSDA(Lorenz63CSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return params.action_perms[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(params.action_perms))
