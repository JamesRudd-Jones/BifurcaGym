import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple
import chex


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray


@struct.dataclass
class EnvParams:
    u_param: float = 0.9  # 18
    k: float = 0.4
    p: float = 6.0

    max_control: float = 0.1  # perturbation to u_param
    maximum_max_control: float = struct.field(False, default=0.1)  # maximum to ensure correct scaling
    u_min: float = 0.0
    u_max: float = 0.99  # keep in typical stable range


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

        self.horizon: int = 200
        self.max_steps_in_ep: int = 2000

        self.periodic_dim: chex.Array = jnp.array((0, 0, 1, 0))  # TODO is this the best way?

        self.action_array: chex.Array = jnp.array((0.0, 1.0, 0.5, -1.0, -0.5))

        self.requires_float64: bool = True

        self.start_point: chex.Array = jnp.array([0.1, 0.1], dtype=jnp.float64)
        self.random_start: bool = False
        self.random_start_low: chex.Array = jnp.array([-2.0, -2.0], dtype=jnp.float64)
        self.random_start_high: chex.Array = jnp.array([2.0, 2.0], dtype=jnp.float64)
        self.fixed_point: chex.Array = jnp.array((0.532754622941, 0.246896772711), dtype=jnp.float64)
        self.reward_ball: float = 1e-3

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

        u_eff = jnp.clip(params.u_param + action, params.u_min, params.u_max)

        r2 = 1.0 + state.x * state.x + state.y * state.y
        tau = params.k - params.p / r2

        ct = jnp.cos(tau)
        st = jnp.sin(tau)

        new_x = 1.0 + u_eff * (state.x * ct - state.y * st)
        new_y = u_eff * (state.x * st + state.y * ct)

        new_state = EnvState(x=new_x, y=new_y, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        obs_new = self.get_obs(new_state)
        obs_old = self.get_obs(state)

        return (jax.lax.stop_gradient(obs_new),
                jax.lax.stop_gradient(obs_new - obs_old),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=self.start_point[0], y=self.start_point[1], time=0)
        rand_start = jrandom.uniform(_key, shape=(2,), minval=self.random_start_low, maxval=self.random_start_high)
        random_state = EnvState(x=rand_start[0], y=rand_start[1], time=0)

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

        diverged = jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y))) > 1e4

        done = jnp.logical_or(jnp.logical_or(state_done, time_done), diverged)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control).squeeze()

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

        margin = 0.5
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
        return "IkedaMap-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(1,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        # The theoretical maximum radius of the Ikeda map is bounded by 1 / (1 - u_max).
        # For u_max = 0.99, the absolute bounds are [-100, 100].
        # This provides perfect min-max normalisation without clipping valid trajectories.
        bound = 1.0 / (1.0 - params.u_max) + 1.0  # Added a tiny buffer

        low = jnp.array([-bound, -bound], dtype=jnp.float64)
        high = jnp.array([bound, bound], dtype=jnp.float64)

        return spaces.Box(low=low, high=high, shape=(2,), dtype=jnp.float64)


class IkedaMapCSDA(IkedaMapCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
