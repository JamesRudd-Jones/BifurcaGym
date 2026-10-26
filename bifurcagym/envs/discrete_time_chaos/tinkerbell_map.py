import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple
import chex
from bifurcagym.envs import utils


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray


@struct.dataclass
class EnvParams:
    a: float = 0.9
    b: float = -0.6013
    c: float = 2.0
    d: float = 0.5

    max_control: float = 0.05  # perturbation to a
    maximum_max_control: float = struct.field(False, default=0.05)  # maximum to ensure correct scaling


class TinkerbellMapCSCA(base_env.BaseEnvironment):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.horizon: int = 200
        self.max_steps_in_ep: int = 2000

        self.action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))

        self.requires_float64: bool = True

        # Canonical initial condition, the basin of attraction is irregular so random starts are kept close to it
        self.start_point: chex.Array = jnp.array([-0.72, -0.64], dtype=jnp.float64)
        self.random_start: bool = True
        self.random_start_low: chex.Array = self.start_point - 0.2
        self.random_start_high: chex.Array = self.start_point + 0.2
        self.reward_ball: float = 0.001
        # Precompute a fixed point (solve map(s) - s = 0)
        self.fixed_point = self._compute_fixed_point(self.default_params)

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

        new_xy = self._map(jnp.array((state.x, state.y)), action, params)
        new_state = EnvState(x=new_xy[0], y=new_xy[1], time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        obs_new = self.get_obs(new_state)
        obs_old = self.get_obs(state)

        return (jax.lax.stop_gradient(obs_new),
                jax.lax.stop_gradient(obs_new - obs_old),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def _map(self, xy: chex.Array, u: chex.Array, params: EnvParams) -> chex.Array:
        x = xy[0]
        y = xy[1]
        xn1 = x * x - y * y + (params.a + u[0]) * x + params.b * y
        yn1 = 2.0 * x * y + (params.c + u[1]) * x + params.d * y

        return jnp.array((xn1, yn1))

    def _compute_fixed_point(self, params: EnvParams) -> chex.Array:
        # Fixed points satisfy map(s, u=0) - s = 0
        def F(xy):
            return self._map(xy, jnp.zeros(2), params) - xy

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

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=self.start_point[0], y=self.start_point[1], time=0)
        rand = jrandom.uniform(_key, shape=(2,), minval=self.random_start_low, maxval=self.random_start_high)
        random_state = EnvState(x=rand[0], y=rand[1], time=0)

        state = jax.tree.map(lambda r, s: jax.lax.select(self.random_start, r, s), random_state, same_state)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        err = jnp.array((state_tp1.x, state_tp1.y)) - self.fixed_point
        reward = -jnp.linalg.norm(err)  # -jnp.sum(err * err)

        state_done = jnp.linalg.norm(err) < self.reward_ball
        time_done = state_tp1.time >= self.max_steps_in_ep

        diverged = jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y))) > 1e3

        done = jnp.logical_or(jnp.logical_or(state_done, time_done), diverged)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x, state.y))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        xs = np.asarray(trajectory_state.x)
        ys = np.asarray(trajectory_state.y)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(self.name)

        margin = 0.2
        ax.set_xlim(np.min(xs) - margin, np.max(xs) + margin)
        ax.set_ylim(np.min(ys) - margin, np.max(ys) + margin)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle='--', alpha=0.6)

        fixed_pt = np.asarray(self.fixed_point)
        ax.plot(fixed_pt[0], fixed_pt[1], 'kx', markersize=10, label="Fixed Point", zorder=3)

        line, = ax.plot([], [], color='blue', linestyle='-', lw=1.0, alpha=0.5, zorder=1)
        dot, = ax.plot([], [], color='red', marker='o', markersize=6, zorder=2)

        ax.legend()

        def update(frame):
            line.set_data(xs[:frame + 1], ys[:frame + 1])

            dot.set_data([xs[frame]], [ys[frame]])

            return line, dot

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=xs.shape[0],
                                       interval=100,
                                       blit=True
                                       )

        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "TinkerbellMap-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(2,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        # TinkerbellMap is unbounded in principle so giving wide bounds  # TODO unsure how to fix this for normalisation
        return spaces.Box(-1e6, 1e6, shape=(2,), dtype=jnp.float64)


class TinkerbellMapCSDA(TinkerbellMapCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: jnp.ndarray = jnp.array(((0.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                                                    (1.0, 0.0), (1.0, 1.0), (1.0, -1.0),
                                                    (-1.0, 0.0), (-1.0, 1.0), (-1.0, -1.0)))

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))