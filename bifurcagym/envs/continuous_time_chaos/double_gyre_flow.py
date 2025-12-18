import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex
import cmocean


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray
    time: int


class DoubleGyreFlowCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # TODO do we need float 64?

        self.x_bounds = jnp.array((0.0, 2.0))
        self.y_bounds = jnp.array((0.0, 1.0))

        self.A = 0.1  # Flow magnitude
        self.epsilon = 0.25  # How much the gyres oscillate
        self.omega = 2.0 * jnp.pi / 10.0  # Frequency

        self.max_speed = 0.05  # slower than peak flow approx 0.3
        self.dt = 0.1

        self.goal_state = jnp.array((1.8, 0.8))

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)  # Action is vx and vy of swimmer

        u_flow, v_flow = self._get_flow_velocity(state.x, state.y, state.time * self.dt)

        new_x = state.x + (u_flow + action[0]) * self.dt  # Euler integration for now
        new_y = state.y + (v_flow + action[1]) * self.dt

        new_x = jnp.clip(new_x, self.x_bounds[0], self.x_bounds[1])
        new_y = jnp.clip(new_y, self.y_bounds[0], self.y_bounds[1])

        new_state = EnvState(x=new_x, y=new_y, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def _get_flow_velocity(self, x: chex.Array, y: chex.Array, t: float) -> Tuple[chex.Array, chex.Array]:
        at = self.epsilon * jnp.sin(self.omega * t)  # Oscillating parameters a(t) and b(t)
        bt = 1 - 2 * self.epsilon * jnp.sin(self.omega * t)

        f = at * x ** 2 + bt * x  # Vel func
        dfdx = 2 * at * x + bt  # Vel derivative

        u_flow = -jnp.pi * self.A * jnp.sin(jnp.pi * f) * jnp.cos(jnp.pi * y)
        v_flow = jnp.pi * self.A * jnp.cos(jnp.pi * f) * jnp.sin(jnp.pi * y) * dfdx

        return u_flow, v_flow

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)
        state = EnvState(x=jnp.array(0.2), y=jnp.array(0.2), time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        dist_to_target = jnp.linalg.norm(jnp.array((state_t.x, state_t.y)) - self.goal_state)
        done = dist_to_target < 0.1

        reward = jnp.array(-0.01)  # Time penalty
        # TODO a goal based reward or some distance metric?

        reward_adder = jax.lax.select(done, 10.0, 0.0)
        reward += reward_adder

        return reward, done

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_speed, self.max_speed)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x, state.y))

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], time=-1)

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fluid_grid_size_plot = 101
        x_fine = jnp.linspace(0, self.x_bounds[1], fluid_grid_size_plot)
        y_fine = jnp.linspace(0, self.y_bounds[1], fluid_grid_size_plot)
        X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine)

        coarse_res = 15  # 25
        x_coarse = jnp.linspace(0, self.x_bounds[1], coarse_res)
        y_coarse = jnp.linspace(0, self.y_bounds[1], coarse_res)
        X_coarse, Y_coarse = jnp.meshgrid(x_coarse, y_coarse)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(self.name)
        ax.set_xlim(float(self.x_bounds[0]), float(self.x_bounds[1]))
        ax.set_ylim(float(self.y_bounds[0]), float(self.y_bounds[1]))
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal', adjustable='box')

        ax.plot(self.goal_state[0], self.goal_state[1], marker='*', markersize=15, color="gold", label="Goal State",
                zorder=4)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

        cmap_val = cmocean.cm.deep_r

        initial_U_coarse, initial_V_coarse = self._get_flow_velocity(X_coarse, Y_coarse, trajectory_state.time[0] * self.dt)
        initial_U_fine, initial_V_fine = self._get_flow_velocity(X_fine, Y_fine, trajectory_state.time[0] * self.dt)
        speed0 = jnp.sqrt(initial_U_fine ** 2 + initial_V_fine ** 2)

        # Setup animation elements
        line, = ax.plot([], [], 'r-', lw=2, label='Agent Trail', zorder=3)
        dot, = ax.plot([], [], color="magenta", marker="o", markersize=10, label='Current State', zorder=5)
        pcm = ax.pcolormesh(X_fine, Y_fine, speed0, cmap=cmap_val, shading='auto', vmin=0,
                            vmax=jnp.max(speed0) * 1.2, zorder=1, alpha=0.7)
        arrow = ax.quiver(X_coarse, Y_coarse, initial_U_coarse, initial_V_coarse, color='black', angles='xy', scale_units='xy',
                          scale=3, width=0.004, zorder=2)
        ax.legend(loc='upper left')

        fig.colorbar(pcm, ax=ax, shrink=0.4, label='Current Magnitude')

        agent_path_x, agent_path_y = [], []

        def update(frame):
            if trajectory_state.time[frame] == 0:
                agent_path_x.clear()
                agent_path_y.clear()

            agent_path_x.append(trajectory_state.x[frame])
            agent_path_y.append(trajectory_state.y[frame])

            line.set_data(agent_path_x, agent_path_y)
            dot.set_data([trajectory_state.x[frame]], [trajectory_state.y[frame]])

            U_coarse, V_coarse = self._get_flow_velocity(X_coarse, Y_coarse, trajectory_state.time[frame] * self.dt)
            U_fine, V_fine = self._get_flow_velocity(X_fine, Y_fine, trajectory_state.time[frame] * self.dt)
            speed = jnp.sqrt(U_fine ** 2 + V_fine ** 2)
            pcm.set_array(speed.ravel())

            arrow.set_UVC(U_coarse, V_coarse)

            return line, dot, pcm, arrow

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(trajectory_state.time),
                                       interval=200,
                                       blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "DoubleGyreFlow-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_speed, self.max_speed, shape=(2,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        lo = jnp.array((self.x_bounds[0], self.y_bounds[0]))
        hi = jnp.array((self.x_bounds[1], self.y_bounds[1]))
        return spaces.Box(lo, hi, (2,), dtype=jnp.float64)


class DoubleGyreFlowCSDA(DoubleGyreFlowCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: chex.Array = jnp.array(((0.0, 0.0),
                                                   (-1.0, 0.0),
                                                   (-1.0, -1.0),
                                                   (0.0, -1.0),
                                                   (1.0, 0.0),
                                                   (1.0, 1.0),
                                                   (0.0, 1.0),
                                                   ))

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action.squeeze()] * self.max_speed

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
