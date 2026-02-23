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
    time: int


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

        # ABC is typically defined on [0, 2π)^3 with periodic boundaries (torus)
        self.L = 2.0 * jnp.pi
        self.x_bounds = jnp.array((0.0, self.L))
        self.y_bounds = jnp.array((0.0, self.L))
        self.z_bounds = jnp.array((0.0, self.L))

        # Flow parameters
        self.A = 1.0
        self.B = 1.0
        self.C = 1.0

        # Control / integration
        self.max_speed = 0.2
        self.dt = 0.05

        # Control objective: reach a goal point on the torus
        self.goal_state = jnp.array((1.7, 4.2, 2.9))  # any point in [0, 2π)^3
        self.goal_radius = 0.25

        self.max_steps_in_episode: int = int(500 // self.dt)

    def step_env(self,
                input_action: Union[jnp.int_, jnp.float_, chex.Array],
                state: EnvState,
                key: chex.PRNGKey,
                ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        u_flow, v_flow, w_flow = self._get_flow_velocity(state.x, state.y, state.z)

        new_x = state.x + (u_flow + action[0]) * self.dt  # TODO do want RK4 or is euler okay for now?
        new_y = state.y + (v_flow + action[1]) * self.dt
        new_z = state.z + (w_flow + action[2]) * self.dt

        # Periodic boundary conditions on [0, 2π)
        new_x = jnp.mod(new_x, self.L)  # TODO can just clip if preferred as saves the torus distance calc
        new_y = jnp.mod(new_y, self.L)
        new_z = jnp.mod(new_z, self.L)

        new_state = EnvState(x=new_x, y=new_y, z=new_z, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        obs_tp1 = self.get_obs(new_state)
        obs_t = self.get_obs(state)

        return (jax.lax.stop_gradient(obs_tp1),
                jax.lax.stop_gradient(obs_tp1 - obs_t),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {},
                )

    def _get_flow_velocity(self, x: chex.Array, y: chex.Array, z: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        # Steady ABC flow (no explicit time dependence)
        u = self.A * jnp.sin(z) + self.C * jnp.cos(y)
        v = self.B * jnp.sin(x) + self.A * jnp.cos(z)
        w = self.C * jnp.sin(y) + self.B * jnp.cos(x)

        return u, v, w

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        state = EnvState(x=jnp.array(0.2), y=jnp.array(0.2), z=jnp.array(0.2), time=0)

        # key_x, key_y, key_z = jrandom.split(key, 3)
        # state = EnvState(x=jrandom.uniform(key_x, (), minval=0.1, maxval=0.3),
        #                  y=jrandom.uniform(key_y, (), minval=0.1, maxval=0.3),
        #                  z=jrandom.uniform(key_z, (), minval=0.1, maxval=0.3),
        #                  time=0)

        return self.get_obs(state), state

    def _torus_delta(self, a: chex.Array, b: chex.Array) -> chex.Array:
        """
        Smallest difference on a circle of length L (for distance on a torus).
        Returns values in [-L/2, L/2].
        """
        d = a - b
        return (d + 0.5 * self.L) % self.L - 0.5 * self.L

    def reward_and_done_function(self,
                                input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                                state_t: EnvState,
                                state_tp1: EnvState,
                                key: chex.PRNGKey = None,
                                ) -> Tuple[chex.Array, chex.Array]:
        # Distance to target on torus with periodic BCs
        dx = self._torus_delta(state_t.x, self.goal_state[0])
        dy = self._torus_delta(state_t.y, self.goal_state[1])
        dz = self._torus_delta(state_t.z, self.goal_state[2])
        dist_to_target = jnp.sqrt(dx * dx + dy * dy + dz * dz)

        reached = dist_to_target < self.goal_radius

        reward = jnp.array(-0.01)  # time penalty
        reward += jax.lax.select(reached, 10.0, 0.0)

        fin_done = jnp.logical_or(reached, state_tp1.time >= self.max_steps_in_episode)

        return reward, fin_done

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        # Continuous control in R^3, clipped componentwise
        return jnp.clip(action, -self.max_speed, self.max_speed)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x, state.y, state.z))

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], z=obs[2], time=-1)

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # 3D coarse grid for vector field visualization
        coarse_res = 8
        x = jnp.linspace(0, self.L, coarse_res)
        y = jnp.linspace(0, self.L, coarse_res)
        z = jnp.linspace(0, self.L, coarse_res)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

        # Initial flow field
        U, V, W = self._get_flow_velocity(X, Y, Z)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_title(self.name)
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_zlim(0, self.L)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Plot goal
        ax.scatter(
            float(self.goal_state[0]),
            float(self.goal_state[1]),
            float(self.goal_state[2]),
            color="gold",
            marker="*",
            s=200,
            label="Goal",
            zorder=5,
        )

        # Plot initial vector field
        quiver = ax.quiver(
            X, Y, Z,
            U, V, W,
            length=0.4,
            normalize=True,
            color="gray",
            alpha=0.5,
        )

        # Agent trajectory elements
        line, = ax.plot([], [], [], "r-", lw=2, label="Agent Trail")
        dot, = ax.plot([], [], [], "mo", markersize=8)

        ax.legend()

        agent_path_x = []
        agent_path_y = []
        agent_path_z = []

        def update(frame):
            if trajectory_state.time[frame] == 0:
                agent_path_x.clear()
                agent_path_y.clear()
                agent_path_z.clear()

            x_val = float(trajectory_state.x[frame])
            y_val = float(trajectory_state.y[frame])
            z_val = float(trajectory_state.z[frame])

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
            frames=len(trajectory_state.time),
            interval=100,
            blit=False,
        )

        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "ABCFlow-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_speed, self.max_speed, shape=(3,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        lo = jnp.array((0.0, 0.0, 0.0))
        hi = jnp.array((self.L, self.L, self.L))
        return spaces.Box(lo, hi, (3,), dtype=jnp.float64)


class ABCFlowCSDA(ABCFlowCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

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
        self.action_opts: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))

        idx = jnp.arange(self.action_opts.shape[0] ** 3)
        powers = self.action_opts.shape[0] ** jnp.arange(3)
        digits = (idx[:, None] // powers[None, :]) % self.action_opts.shape[0]
        self.action_array: jnp.ndarray = self.action_opts[digits]
        # TODO should I add the following to utils to standardise it?

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action.squeeze()] * self.max_speed

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
