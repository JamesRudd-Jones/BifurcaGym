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


@struct.dataclass
class EnvParams:
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7

    max_control: chex.Array = jnp.array((0.01, 0.01, 1.0))
    maximum_max_control: chex.Array = struct.field(False, default=jnp.array((0.01, 0.01, 1.0)))  # maximum to ensure correct scaling

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

        self.dt: float = 0.02
        self.horizon: int = 200
        self.max_steps_in_ep: int = int(200 // self.dt)
        self.num_actions: int = 3
        self.substeps: int = 5

        self.action_array: chex.Array = utils.action_permutations_generation(self.num_actions)

        self.requires_float64: bool = True

        self.start_point: chex.Array = jnp.array([0.1, 0.0, 0.0], dtype=jnp.float64)
        self.random_start: bool = False
        self.random_start_low: chex.Array = jnp.array([-5.0, -5.0, 0.0], dtype=jnp.float64)
        self.random_start_high: chex.Array = jnp.array([5.0, 5.0, 10.0], dtype=jnp.float64)
        self.reward_ball: float = 1e-2

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

        new_x = utils.integrate_ode(self._f, state.time * self.dt, state.x, u, self.dt, self.substeps, params)
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

    def _f(self, t: float, x: chex.Array, u: chex.Array, params: EnvParams) -> chex.Array:
        X, Y, Z = x[0], x[1], x[2]
        dx = -Y - Z
        dy = X + (params.a + u[0]) * Y
        dz = (params.b + u[1]) + Z * (X - (params.c + u[2]))
        # dy = X + self.a * Y
        # dz = self.b + Z * (X - (self.c + u[0]))
        return jnp.array([dx, dy, dz], dtype=jnp.float64)

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        same_state = EnvState(x=self.start_point, time=0)
        rand_x = jrandom.uniform(_key, shape=(3,), minval=self.random_start_low, maxval=self.random_start_high)
        random_state = EnvState(x=rand_x, time=0)
        state = jax.tree.map(lambda r, s: jax.lax.select(self.random_start, r, s), random_state, same_state)

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

        state_done = jnp.linalg.norm(err) < self.reward_ball
        time_done = state_tp1.time >= self.max_steps_in_ep
        boundary_done = jnp.linalg.norm(state_tp1.x) > 1e3
        done = jnp.logical_or(jnp.logical_or(state_done, boundary_done), time_done)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control).squeeze()

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.asarray(state.x)

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(x=jnp.asarray(obs), time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        states = np.asarray(trajectory_state.x)
        xs = states[:, 0]
        ys = states[:, 1]
        zs = states[:, 2]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.name)

        margin = 2.0
        ax.set_xlim(np.min(xs) - margin, np.max(xs) + margin)
        ax.set_ylim(np.min(ys) - margin, np.max(ys) + margin)
        ax.set_zlim(np.min(zs) - margin, np.max(zs) + margin)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        fixed_pt = np.asarray(params.fixed_point)
        ax.plot([fixed_pt[0, 0]], [fixed_pt[1, 0]], [fixed_pt[2, 0]], 'kx', markersize=8, label="Fixed Point")

        line, = ax.plot([], [], [], color='blue', linestyle='-', lw=1.0, alpha=0.8)
        dot, = ax.plot([], [], [], color='red', marker='o', markersize=6)

        ax.legend()

        def update(frame):
            line.set_data(xs[:frame], ys[:frame])
            line.set_3d_properties(zs[:frame])

            dot.set_data([xs[frame]], [ys[frame]])
            dot.set_3d_properties([zs[frame]])

            return line, dot

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=states.shape[0],
                                       interval=self.dt * 1000,  # Convert dt to milliseconds
                                       blit=False
                                       )

        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "Rossler-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, shape=(self.num_actions,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        # Rossler is unbounded in principle so giving wide bounds  # TODO unsure how to fix this for normalisation
        return spaces.Box(-1e6, 1e6, shape=(3,), dtype=jnp.float64)


class RosslerCSDA(RosslerCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))