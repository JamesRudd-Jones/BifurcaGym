import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex


@struct.dataclass
class LorenzEnvState(base_env.EnvState):
    x: jnp.ndarray
    time: int


class LorenzCSCA(base_env.BaseEnvironment):
    """
    Lorenz system:
      x' = sigma (y - x)
      y' = x (rho - z) - y
      z' = x y - beta z

    Control u perturbs rho: rho_eff = rho + u (clipped).
    """

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # canonical chaotic params
        self.sigma: float = 10.0
        self.rho: float = 28.0
        self.beta: float = 8.0 / 3.0

        # integration
        self.dt: float = 0.01
        self.substeps: int = 5

        # control
        self.max_control: float = 5.0  # rho perturbation bound

        # episode
        self.max_steps_in_episode: int = int(500 // self.dt)  # with dt=0.01 then there are 50 time units
        self.reward_ball: float = 1e-2

        # init
        self.start_point = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)
        self.random_start: bool = True
        self.random_start_low = jnp.array([-10.0, -10.0, 0.0], dtype=jnp.float64)
        self.random_start_high = jnp.array([10.0, 10.0, 30.0], dtype=jnp.float64)

        # (optional) horizon for algorithms that use it
        self.horizon: int = 200

    @property
    def fixed_point(self) -> chex.Array:
        """
        Lorenz equilibria:
          (0,0,0)
          (±sqrt(beta*(rho-1)), ±sqrt(beta*(rho-1)), rho-1) for rho>1
        For rho=28, origin is unstable; nonzero equilibria are also unstable.
        We'll target the origin by default (simple, classic).
        """
        return jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)

    def _f(self, x: chex.Array, u: chex.Array) -> chex.Array:
        sigma, beta = self.sigma, self.beta
        rho_eff = self.rho + u  # control perturbs rho
        X, Y, Z = x[0], x[1], x[2]
        dx = sigma * (Y - X)
        dy = X * (rho_eff - Z) - Y
        dz = X * Y - beta * Z
        return jnp.array([dx, dy, dz], dtype=jnp.float64)

    def action_convert(self, action: Union[jnp.int_, jnp.float_, chex.Array]) -> chex.Array:
        a = jnp.asarray(action).squeeze()
        a = jnp.clip(a, -self.max_control, self.max_control)
        return a

    def step_env(
        self,
        input_action: Union[jnp.int_, jnp.float_, chex.Array],
        state: LorenzEnvState,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, LorenzEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        u = self.action_convert(input_action)

        new_x = integrate_ode(self._f, state.x, u, self.dt, self.substeps)
        new_state = LorenzEnvState(x=new_x, time=state.time + 1)

        reward, done = self.reward_and_done_function(state, new_state)

        obs_new = self.get_obs(new_state)
        obs_old = self.get_obs(state)

        return (
            jax.lax.stop_gradient(obs_new),
            jax.lax.stop_gradient(obs_new - obs_old),
            jax.lax.stop_gradient(new_state),
            reward,
            done,
            {},
        )

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, LorenzEnvState]:
        key, k = jrandom.split(key)
        same_state = LorenzEnvState(x=self.start_point, time=0)
        rand_x = jrandom.uniform(k, shape=(3,), minval=self.random_start_low, maxval=self.random_start_high)
        random_state = LorenzEnvState(x=rand_x, time=0)
        state = jax.tree.map(lambda r, s: jax.lax.select(self.random_start, r, s), random_state, same_state)
        return self.get_obs(state), state

    def reward_and_done_function(
        self,
        state_t: LorenzEnvState,
        state_tp1: LorenzEnvState,
    ) -> Tuple[chex.Array, chex.Array]:
        err = state_tp1.x - self.fixed_point
        reward = -jnp.sum(err * err)

        state_done = jnp.linalg.norm(err) < self.reward_ball
        time_done = state_tp1.time >= self.max_steps_in_episode
        done = jnp.logical_or(state_done, time_done)
        return reward, done

    def get_obs(self, state: LorenzEnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.asarray(state.x)

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> LorenzEnvState:
        return LorenzEnvState(x=jnp.asarray(obs), time=-1)

    @property
    def name(self) -> str:
        return "Lorenz-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, shape=(1,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        # Lorenz is unbounded in principle; give wide bounds
        return spaces.Box(-1e6, 1e6, shape=(3,), dtype=jnp.float64)


# ============================================================
# 2) Rössler environment (continuous-time, continuous action)
# Control: action perturbs c: c_eff = c + u
# Target: stabilize an unstable equilibrium (for standard params)
# ============================================================

@struct.dataclass
class RosslerEnvState(base_env.EnvState):
    x: jnp.ndarray  # shape (3,)
    time: int


class RosslerCSCA(base_env.BaseEnvironment):
    """
    Rössler system:
      x' = -y - z
      y' = x + a y
      z' = b + z (x - c)

    Control u perturbs c: c_eff = c + u (clipped).
    """

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # common chaotic params
        self.a: float = 0.2
        self.b: float = 0.2
        self.c: float = 5.7

        # integration
        self.dt: float = 0.02
        self.substeps: int = 5

        # control
        self.max_control: float = 1.0  # c perturbation bound

        # episode
        self.max_steps_in_episode: int = 3000
        self.reward_ball: float = 1e-2

        # init
        self.start_point = jnp.array([0.1, 0.0, 0.0], dtype=jnp.float64)
        self.random_start: bool = True
        self.random_start_low = jnp.array([-5.0, -5.0, 0.0], dtype=jnp.float64)
        self.random_start_high = jnp.array([5.0, 5.0, 10.0], dtype=jnp.float64)

        self.horizon: int = 200

    @property
    def fixed_point(self) -> chex.Array:
        """
        Equilibria solve:
          -y - z = 0  => y = -z
          x + a y = 0 => x = -a y = a z
          b + z(x - c) = 0 => b + z(a z - c) = 0 => a z^2 - c z + b = 0

        Choose the "+" root (often the one relevant in chaos-control demos), but you can swap if desired.
        """
        a, b, c = self.a, self.b, self.c
        disc = c * c - 4.0 * a * b
        # guard against numerical negatives
        disc = jnp.maximum(disc, 0.0)
        z = (c + jnp.sqrt(disc)) / (2.0 * a)
        x = a * z
        y = -z
        return jnp.array([x, y, z], dtype=jnp.float64)

    def _f(self, x: chex.Array, u: chex.Array) -> chex.Array:
        a, b = self.a, self.b
        c_eff = self.c + u
        X, Y, Z = x[0], x[1], x[2]
        dx = -Y - Z
        dy = X + a * Y
        dz = b + Z * (X - c_eff)
        return jnp.array([dx, dy, dz], dtype=jnp.float64)

    def action_convert(self, action: Union[jnp.int_, jnp.float_, chex.Array]) -> chex.Array:
        a = jnp.asarray(action).squeeze()
        a = jnp.clip(a, -self.max_control, self.max_control)
        return a

    def step_env(
        self,
        input_action: Union[jnp.int_, jnp.float_, chex.Array],
        state: RosslerEnvState,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, RosslerEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        u = self.action_convert(input_action)

        new_x = integrate_ode(self._f, state.x, u, self.dt, self.substeps)
        new_state = RosslerEnvState(x=new_x, time=state.time + 1)

        reward, done = self.reward_and_done_function(state, new_state)

        obs_new = self.get_obs(new_state)
        obs_old = self.get_obs(state)

        return (
            jax.lax.stop_gradient(obs_new),
            jax.lax.stop_gradient(obs_new - obs_old),
            jax.lax.stop_gradient(new_state),
            reward,
            done,
            {},
        )

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, RosslerEnvState]:
        key, k = jrandom.split(key)
        same_state = RosslerEnvState(x=self.start_point, time=0)
        rand_x = jrandom.uniform(k, shape=(3,), minval=self.random_start_low, maxval=self.random_start_high)
        random_state = RosslerEnvState(x=rand_x, time=0)
        state = jax.tree.map(lambda r, s: jax.lax.select(self.random_start, r, s), random_state, same_state)
        return self.get_obs(state), state

    def reward_and_done_function(
        self,
        state_t: RosslerEnvState,
        state_tp1: RosslerEnvState,
    ) -> Tuple[chex.Array, chex.Array]:
        err = state_tp1.x - self.fixed_point
        reward = -jnp.sum(err * err)

        state_done = jnp.linalg.norm(err) < self.reward_ball
        time_done = state_tp1.time >= self.max_steps_in_episode
        done = jnp.logical_or(state_done, time_done)
        return reward, done

    def get_obs(self, state: RosslerEnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.asarray(state.x)

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> RosslerEnvState:
        return RosslerEnvState(x=jnp.asarray(obs), time=-1)

    @property
    def name(self) -> str:
        return "Rossler-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, shape=(1,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(-1e6, 1e6, shape=(3,), dtype=jnp.float64)


# ============================================================
# 3) Chua environment (continuous-time, continuous action)
# Classic 3D Chua's circuit (dimensionless form)
# Control: action perturbs alpha: alpha_eff = alpha + u
# Target: stabilize an equilibrium (choose origin by default)
# ============================================================

@struct.dataclass
class ChuaEnvState(base_env.EnvState):
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

    Note: Parameterizations vary across papers. This one is a standard choice for the double-scroll regime.
    """

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # typical double-scroll-ish set (dimensionless)
        self.alpha: float = 15.6
        self.beta: float = 28.0
        self.m0: float = -1.143
        self.m1: float = -0.714

        # integration
        self.dt: float = 0.01
        self.substeps: int = 5

        # control
        self.max_control: float = 5.0  # alpha perturbation bound

        # episode
        self.max_steps_in_episode: int = 5000
        self.reward_ball: float = 1e-2

        # init
        self.start_point = jnp.array([0.1, 0.0, 0.0], dtype=jnp.float64)
        self.random_start: bool = True
        self.random_start_low = jnp.array([-2.0, -2.0, -2.0], dtype=jnp.float64)
        self.random_start_high = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float64)

        self.horizon: int = 200

    def _f_nl(self, x: chex.Array) -> chex.Array:
        """Chua piecewise-linear diode characteristic f(x)."""
        m0, m1 = self.m0, self.m1
        return m1 * x + 0.5 * (m0 - m1) * (jnp.abs(x + 1.0) - jnp.abs(x - 1.0))

    @property
    def fixed_point(self) -> chex.Array:
        """
        Equilibria satisfy:
          y - x - f(x) = 0
          x - y + z = 0
          y = 0  (since z'=-beta y must be 0 at equilibrium)
        => y = 0
           z = -x
           and 0 - x - f(x) = 0 => x + f(x) = 0

        x=0 is always a solution (since f(0)=m1*0 + ... = 0). We'll target origin.
        """
        return jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)

    def _f(self, x: chex.Array, u: chex.Array) -> chex.Array:
        alpha_eff = self.alpha + u
        beta = self.beta
        X, Y, Z = x[0], x[1], x[2]
        fx = self._f_nl(X)
        dx = alpha_eff * (Y - X - fx)
        dy = X - Y + Z
        dz = -beta * Y
        return jnp.array([dx, dy, dz], dtype=jnp.float64)

    def action_convert(self, action: Union[jnp.int_, jnp.float_, chex.Array]) -> chex.Array:
        a = jnp.asarray(action).squeeze()
        a = jnp.clip(a, -self.max_control, self.max_control)
        return a

    def step_env(
        self,
        input_action: Union[jnp.int_, jnp.float_, chex.Array],
        state: ChuaEnvState,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, ChuaEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        u = self.action_convert(input_action)

        new_x = integrate_ode(self._f, state.x, u, self.dt, self.substeps)
        new_state = ChuaEnvState(x=new_x, time=state.time + 1)

        reward, done = self.reward_and_done_function(state, new_state)

        obs_new = self.get_obs(new_state)
        obs_old = self.get_obs(state)

        return (
            jax.lax.stop_gradient(obs_new),
            jax.lax.stop_gradient(obs_new - obs_old),
            jax.lax.stop_gradient(new_state),
            reward,
            done,
            {},
        )

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, ChuaEnvState]:
        key, k = jrandom.split(key)
        same_state = ChuaEnvState(x=self.start_point, time=0)
        rand_x = jrandom.uniform(k, shape=(3,), minval=self.random_start_low, maxval=self.random_start_high)
        random_state = ChuaEnvState(x=rand_x, time=0)
        state = jax.tree.map(lambda r, s: jax.lax.select(self.random_start, r, s), random_state, same_state)
        return self.get_obs(state), state

    def reward_and_done_function(
        self,
        state_t: ChuaEnvState,
        state_tp1: ChuaEnvState,
    ) -> Tuple[chex.Array, chex.Array]:
        err = state_tp1.x - self.fixed_point
        reward = -jnp.sum(err * err)

        state_done = jnp.linalg.norm(err) < self.reward_ball
        time_done = state_tp1.time >= self.max_steps_in_episode
        done = jnp.logical_or(state_done, time_done)
        return reward, done

    def get_obs(self, state: ChuaEnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.asarray(state.x)

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> ChuaEnvState:
        return ChuaEnvState(x=jnp.asarray(obs), time=-1)

    @property
    def name(self) -> str:
        return "Chua-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, shape=(1,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(-1e6, 1e6, shape=(3,), dtype=jnp.float64)
