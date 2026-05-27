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
    init_a: float = 1.4
    init_b: float = 0.3

    max_control: float = 0.1
    maximum_max_control: float = struct.field(False, default=0.1)  # maximum to ensure correct scaling


class HenonMapCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.discretisation: int = 100 + 1
        self.horizon: int = 200
        self.max_steps_in_ep: int = 1000
        self.ref_vector = jnp.linspace(0, 1, self.discretisation)

        self.action_array: chex.Array = jnp.array(((0.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                                                   (1.0, 0.0), (1.0, 1.0), (1.0, -1.0),
                                                   (-1.0, 0.0), (-1.0, 1.0), (-1.0, -1.0)))

        self.requires_float64: bool = True

        self.start_point: float = 0.0
        self.random_start: bool = True
        self.random_start_range_lower: float = -1.4
        self.random_start_range_upper: float = 1.4
        self.fixed_point: chex.Array = jnp.array((0.631354477, 0.189406343), dtype=jnp.float64)
        self.reward_ball: float = 0.001

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

        new_x = 1 - (params.init_a + action[0]) * jnp.square(state.x) + state.y

        new_y = (params.init_b + action[1]) * state.x

        new_state = EnvState(x=new_x, y=new_y, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
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

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        err = jnp.array((state_tp1.x, state_tp1.y)) - self.fixed_point
        reward = -jnp.linalg.norm(err, 2)
        # the above can set more specific norm distances

        # TODO state_t or state_tp1
        # boundary_done_x = jnp.logical_or(state_tp1.x <= -10, state_tp1.x >= 10)
        # boundary_done_y = jnp.logical_or(state_tp1.y <= -10, state_tp1.y >= 10)
        # boundary_done = jnp.logical_or(boundary_done_x, boundary_done_y)
        boundary_done = jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y))) > 1e3
        goal_done = jnp.linalg.norm(err) < self.reward_ball

        done = jnp.logical_or(jnp.logical_or(boundary_done, goal_done), state_tp1.time >= self.max_steps_in_ep)

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
        return "HenonMap-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(2,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-1.4, 1.4, (2,), dtype=jnp.float64)


class HenonMapCSDA(HenonMapCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))


class HenonMapDSDA(HenonMapCSDA):
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

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((self._projection(state.x), self._projection(state.y)))

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.discretisation)
