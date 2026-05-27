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


@struct.dataclass
class EnvParams:
    init_mu: float = 2

    max_control: float = 0.1
    maximum_max_control: float = struct.field(False, default=0.1)  # maximum to ensure correct scaling

    @property
    def fixed_point(self) -> float:
        return self.init_mu / (self.init_mu + 1)


class TentMapCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.discretisation: int = 100 + 1
        self.horizon: int = 200
        self.max_steps_in_ep: int = 500
        self.ref_vector = jnp.linspace(0, 1, self.discretisation)

        self.action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))

        self.requires_float64: bool = True

        self.start_point: float = 0.1
        self.random_start: bool = True
        self.random_start_range_lower: float = 0.0
        self.random_start_range_upper: float = 1.0
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

        new_x = (action + params.init_mu) * jnp.minimum(state.x, 1 - state.x)

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
        same_state = EnvState(x=jnp.array(self.start_point), time=0)
        random_state = EnvState(x=jrandom.uniform(_key,
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
        reward = -jnp.abs(state_tp1.x - params.fixed_point) ** 2
        # The above can set more specific norm distances

        # TODO state_t or state_tp1
        done = jnp.logical_or(jnp.abs(state_tp1.x - params.fixed_point) < self.reward_ball, state_tp1.time >= self.max_steps_in_ep)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control).squeeze()

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x,))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(x=obs[0], time=0)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        xs = np.asarray(trajectory_state.x).flatten()
        ts = np.arange(len(xs))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(self.name)

        ax.set_xlim(0, len(xs))
        ax.set_ylim(-0.1, 1.1)

        ax.set_xlabel("Time Step (n)")
        ax.set_ylabel("State (x)")
        ax.grid(True, linestyle='--', alpha=0.6)

        fixed_pt_array = np.asarray(params.fixed_point).flatten()
        fixed_pt = float(fixed_pt_array[0]) if len(fixed_pt_array) > 0 else 0.0
        ax.axhline(fixed_pt, color='k', linestyle='--', lw=1.5, label="Fixed Point", zorder=1)

        line, = ax.plot([], [], color='blue', linestyle='-', lw=1.0, alpha=0.7, marker='.', markersize=4, zorder=2)
        dot, = ax.plot([], [], color='red', marker='o', markersize=6, zorder=3)

        ax.legend()

        def update(frame):
            line.set_data(ts[:frame + 1], xs[:frame + 1])
            dot.set_data([ts[frame]], [xs[frame]])
            return line, dot

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(xs),
                                       interval=100,
                                       blit=True
                                       )

        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "TentMap-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(1,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (1,), dtype=jnp.float64)


class TentMapCSDA(TentMapCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))


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

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.discretisation)