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
    z: jnp.ndarray


@struct.dataclass
class EnvParams:
    # # 7 actions: noop and rook moves, no diagonals for scalability as of now
    # self.action_array: chex.Array = jnp.array(((0.0, 0.0, 0.0),
    #                                            (1.0, 0.0, 0.0),
    #                                            (-1.0, 0.0, 0.0),
    #                                            (0.0, 1.0, 0.0),
    #                                            (0.0, -1.0, 0.0),
    #                                            (0.0, 0.0, 1.0),
    #                                            (0.0, 0.0, -1.0),
    #                                            ))

    # diagonals
    action_opts: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))

    idx = jnp.arange(action_opts.shape[0] ** 3)
    powers = action_opts.shape[0] ** jnp.arange(3)
    digits = (idx[:, None] // powers[None, :]) % action_opts.shape[0]
    action_array: chex.Array = struct.field(False, default=action_opts[digits])
    # TODO should I add the following to utils to standardise it?

    dt: float = struct.field(False, default=0.05)

    # ABC is typically defined on [0, 2π)^3 with periodic boundaries (torus)
    L: float = struct.field(False, default = 2.0 * jnp.pi)

    goal_state: chex.Array = struct.field(False, default=jnp.array((1.7, 4.2, 2.9)))  # any point in [0, 2π)^3
    goal_radius: float = struct.field(False, default=0.25)

    A: float = 1.0
    B: float = 1.0
    C: float = 1.0

    max_speed: float = 0.2
    maximum_max_speed: float = struct.field(False, default=0.2)  # maximum to ensure correct scaling

    @property
    def max_steps_in_ep(self) -> int:
        return int(500 // self.dt)

    @property
    def x_bounds(self) -> chex.Array:
        return struct.field(False, default=jnp.array((0.0, self.L)))

    @property
    def y_bounds(self) -> chex.Array:
        return struct.field(False, default=jnp.array((0.0, self.L)))

    @property
    def z_bounds(self) -> chex.Array:
        return struct.field(False, default=jnp.array((0.0, self.L)))


class ABCFlowCSCA(base_env.BaseEnvironment):
    """
    Arnold–Beltrami–Childress (ABC) flow on a 3D torus, with additive control.

    Standard steady ABC flow:
        u = A sin(z) + C cos(y)
        v = B sin(x) + A cos(z)
        w = C sin(y) + B cos(x)

    Control action is an additive velocity (ux, uy, uz) clipped to max_speed.
    State evolves via Euler integration.
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
        action = self.action_convert(input_action, params)

        u_flow, v_flow, w_flow = self._get_flow_velocity(state.x, state.y, state.z, params)

        new_x = state.x + (u_flow + action[0]) * params.dt  # TODO do want RK4 or is euler okay for now?
        new_y = state.y + (v_flow + action[1]) * params.dt
        new_z = state.z + (w_flow + action[2]) * params.dt

        # Periodic boundary conditions on [0, 2π)
        new_x = jnp.mod(new_x, params.L)  # TODO can just clip if preferred as saves the torus distance calc
        new_y = jnp.mod(new_y, params.L)
        new_z = jnp.mod(new_z, params.L)

        new_state = EnvState(x=new_x, y=new_y, z=new_z, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        obs_tp1 = self.get_obs(new_state)
        obs_t = self.get_obs(state)

        return (jax.lax.stop_gradient(obs_tp1),
                jax.lax.stop_gradient(obs_tp1 - obs_t),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {},
                )

    def _get_flow_velocity(self, x: chex.Array, y: chex.Array, z: chex.Array, params: EnvParams) -> Tuple[chex.Array, chex.Array, chex.Array]:
        # Steady ABC flow (no explicit time dependence)
        u = params.A * jnp.sin(z) + params.C * jnp.cos(y)
        v = params.B * jnp.sin(x) + params.A * jnp.cos(z)
        w = params.C * jnp.sin(y) + params.B * jnp.cos(x)

        return u, v, w

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # state = EnvState(x=jnp.array(0.2), y=jnp.array(0.2), z=jnp.array(0.2), time=0)

        key_x, key_y, key_z = jrandom.split(key, 3)
        delta = 0.001
        start_point = 0.2
        state = EnvState(x=jrandom.uniform(key_x, (), minval=0.2 - delta, maxval=0.2 + delta),
                         y=jrandom.uniform(key_y, (), minval=0.2 - delta, maxval=0.2 + delta),
                         z=jrandom.uniform(key_z, (), minval=0.2 - delta, maxval=0.2 + delta),
                         time=0)

        return self.get_obs(state), state

    def _torus_delta(self, a: chex.Array, b: chex.Array, params: EnvParams) -> chex.Array:
        """
        Smallest difference on a circle of length L (for distance on a torus).
        Returns values in [-L/2, L/2].
        """
        d = a - b
        return (d + 0.5 * params.L) % params.L - 0.5 * params.L

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        # Distance to target on torus with periodic BCs
        dx = self._torus_delta(state_t.x, params.goal_state[0], params)
        dy = self._torus_delta(state_t.y, params.goal_state[1], params)
        dz = self._torus_delta(state_t.z, params.goal_state[2], params)
        dist_to_target = jnp.sqrt(dx * dx + dy * dy + dz * dz)

        reached = dist_to_target < params.goal_radius

        reward = jnp.array(-0.01)  # time penalty
        reward += jax.lax.select(reached, 10.0, 0.0)

        fin_done = jnp.logical_or(reached, state_tp1.time >= params.max_steps_in_ep)

        return reward, fin_done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        # Continuous control in R^3, clipped componentwise
        return jnp.clip(action, -params.max_speed, params.max_speed)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x, state.y, state.z))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], z=obs[2], time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        times = np.asarray(trajectory_state.time)
        xs = np.asarray(trajectory_state.x)
        ys = np.asarray(trajectory_state.y)
        zs = np.asarray(trajectory_state.z)

        # coarse_res = 8
        # x = np.linspace(0, params.L, coarse_res)
        # y = np.linspace(0, params.L, coarse_res)
        # z = np.linspace(0, params.L, coarse_res)
        # X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        #
        # U, V, W = self._get_flow_velocity(X, Y, Z)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_title(self.name)
        ax.set_xlim(0, params.L)
        ax.set_ylim(0, params.L)
        ax.set_zlim(0, params.L)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.scatter(float(params.goal_state[0]),
                   float(params.goal_state[1]),
                   float(params.goal_state[2]),
                   color="gold",
                   marker="*",
                   s=200,
                   label="Goal",
                   zorder=5,
                   )

        # quiver = ax.quiver(
        #     X, Y, Z,
        #     U, V, W,
        #     length=0.4,
        #     normalize=True,
        #     color="gray",
        #     alpha=0.5,
        # )

        line, = ax.plot([], [], [], "r-", lw=2, label="Agent Trail")
        dot, = ax.plot([], [], [], "mo", markersize=8)

        ax.legend()

        agent_path_x = []
        agent_path_y = []
        agent_path_z = []

        def update(frame):
            if times[frame] == 0:
                agent_path_x.clear()
                agent_path_y.clear()
                agent_path_z.clear()

            x_val = float(xs[frame])
            y_val = float(ys[frame])
            z_val = float(zs[frame])

            agent_path_x.append(x_val)
            agent_path_y.append(y_val)
            agent_path_z.append(z_val)

            line.set_data(agent_path_x, agent_path_y)
            line.set_3d_properties(agent_path_z)

            dot.set_data([x_val], [y_val])
            dot.set_3d_properties([z_val])

            return line, dot

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=100,
            blit=False,
        )

        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "ABCFlow-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_speed, params.maximum_max_speed, shape=(3,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        lo = jnp.array((0.0, 0.0, 0.0))
        hi = jnp.array((params.L, params.L, params.L))
        return spaces.Box(lo, hi, (3,), dtype=jnp.float64)


class ABCFlowCSDA(ABCFlowCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Array, params: EnvParams) -> chex.Numeric:
        return params.action_array[action.squeeze()] * params.max_speed

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(params.action_array))
