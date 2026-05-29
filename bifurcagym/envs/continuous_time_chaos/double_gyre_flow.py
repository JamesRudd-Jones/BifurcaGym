import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env, utils
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple
import chex
import cmocean


@struct.dataclass
class EnvState(base_env.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray


@struct.dataclass
class EnvParams:
    A: float = 0.1  # Flow magnitude
    epsilon: float = 0.25  # How much the gyres oscillate
    omega: float = 2.0 * jnp.pi / 10.0  # Frequency

    max_speed: float = 0.05  # slower than peak flow approx 0.3
    maximum_max_speed: float = struct.field(False, default=0.05)  # maximum to ensure correct scaling


class DoubleGyreFlowCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = 0.1
        self.horizon: int = 25
        self.max_steps_in_ep: int = int(200 // self.dt)

        self.action_array: chex.Array = jnp.array(((0.0, 0.0),
                                                    (-1.0, 0.0),
                                                    (-1.0, -1.0),
                                                    (0.0, -1.0),
                                                    (1.0, 0.0),
                                                    (1.0, 1.0),
                                                    (0.0, 1.0),
                                                    ))

        self.requires_float64: bool = True

        self.x_bounds: chex.Array = jnp.array((0.0, 2.0))
        self.y_bounds: chex.Array = jnp.array((0.0, 1.0))
        self.goal_state: chex.Array = jnp.array((1.8, 0.8))

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(self,
                 input_action: chex.Numeric,
                 state: EnvState,
                 params: EnvParams,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action, params)  # Action is vx and vy of swimmer

        state_arr = jnp.array((state.x, state.y))

        def dynamics(t_curr, x_arr, u_arr, p):
            u_flow, v_flow = self._get_flow_velocity(x_arr[0], x_arr[1], t_curr, p)
            return jnp.array((u_flow + u_arr[0], v_flow + u_arr[1]))

        new_state_arr = utils.rk4_step(dynamics, state.time * self.dt, state_arr, action, self.dt, params)

        new_x = new_state_arr[0]
        new_y = new_state_arr[1]

        new_x = jnp.clip(new_x, self.x_bounds[0], self.x_bounds[1])
        new_y = jnp.clip(new_y, self.y_bounds[0], self.y_bounds[1])

        new_state = EnvState(x=new_x, y=new_y, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        obs_tp1 = self.get_obs(new_state)
        obs = self.get_obs(state)

        return (jax.lax.stop_gradient(obs_tp1),
                jax.lax.stop_gradient(obs_tp1 - obs),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def _get_flow_velocity(self, x: chex.Array, y: chex.Array, t: float, params: EnvParams) -> Tuple[chex.Array, chex.Array]:
        at = params.epsilon * jnp.sin(params.omega * t)  # Oscillating parameters a(t) and b(t)
        bt = 1 - 2 * params.epsilon * jnp.sin(params.omega * t)

        f = at * x ** 2 + bt * x  # Vel func
        dfdx = 2 * at * x + bt  # Vel derivative

        u_flow = -jnp.pi * params.A * jnp.sin(jnp.pi * f) * jnp.cos(jnp.pi * y)
        v_flow = jnp.pi * params.A * jnp.cos(jnp.pi * f) * jnp.sin(jnp.pi * y) * dfdx

        return u_flow, v_flow

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # key, _key = jrandom.split(key)
        state = EnvState(x=jnp.array(0.2), y=jnp.array(0.2), time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        dist_to_target = jnp.linalg.norm(jnp.array((state_t.x, state_t.y)) - self.goal_state)
        done = dist_to_target < 0.1

        reward = jnp.array(-0.01)  # Time penalty
        # TODO a goal based reward or some distance metric?
        reward += jax.lax.select(done, 10.0, 0.0)

        fin_done = jnp.logical_or(done, state_tp1.time >= self.max_steps_in_ep)

        return reward, fin_done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_speed, params.max_speed)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array((state.x, state.y, state.time))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], time=obs[2])

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        times = np.asarray(trajectory_state.time)
        xs = np.asarray(trajectory_state.x)
        ys = np.asarray(trajectory_state.y)

        fluid_grid_size_plot = 101
        x_fine = np.linspace(0, self.x_bounds[1], fluid_grid_size_plot)
        y_fine = np.linspace(0, self.y_bounds[1], fluid_grid_size_plot)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

        coarse_res = 15  # 25
        x_coarse = np.linspace(0, self.x_bounds[1], coarse_res)
        y_coarse = np.linspace(0, self.y_bounds[1], coarse_res)
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

        def get_curr_param(params, time_idx):
            return jax.tree.map(lambda x: x[time_idx] if getattr(x, 'ndim', 0) > 0 else x, params)

        initial_U_coarse, initial_V_coarse = self._get_flow_velocity(X_coarse, Y_coarse, times[0] * self.dt, get_curr_param(params, 0))
        initial_U_fine, initial_V_fine = self._get_flow_velocity(X_fine, Y_fine, times[0] * self.dt, get_curr_param(params, 0))
        speed0 = np.sqrt(initial_U_fine ** 2 + initial_V_fine ** 2)

        # Setup animation elements
        line, = ax.plot([], [], 'r-', lw=2, label='Agent Trail', zorder=3)
        dot, = ax.plot([], [], color="magenta", marker="o", markersize=10, label='Current State', zorder=5)
        pcm = ax.pcolormesh(X_fine, Y_fine, speed0, cmap=cmap_val, shading='auto', vmin=0,
                            vmax=np.max(speed0) * 1.2, zorder=1, alpha=0.7)
        arrow = ax.quiver(X_coarse, Y_coarse, initial_U_coarse, initial_V_coarse, color='black', angles='xy', scale_units='xy',
                          scale=3, width=0.004, zorder=2)
        ax.legend(loc='upper left')

        fig.colorbar(pcm, ax=ax, shrink=0.4, label='Current Magnitude')

        agent_path_x, agent_path_y = [], []

        def update(frame):
            if trajectory_state.time[frame] == 0:
                agent_path_x.clear()
                agent_path_y.clear()

            agent_path_x.append(xs[frame])
            agent_path_y.append(ys[frame])

            line.set_data(agent_path_x, agent_path_y)
            dot.set_data([xs[frame]], [ys[frame]])

            U_coarse, V_coarse = self._get_flow_velocity(X_coarse, Y_coarse, times[frame] * self.dt, get_curr_param(params, frame))
            U_fine, V_fine = self._get_flow_velocity(X_fine, Y_fine, times[frame] * self.dt, get_curr_param(params, frame))
            speed = jnp.sqrt(U_fine ** 2 + V_fine ** 2)
            pcm.set_array(speed.ravel())

            arrow.set_UVC(U_coarse, V_coarse)

            return line, dot, pcm, arrow

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(times),
                                       interval=200,
                                       blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "DoubleGyreFlow-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_speed, params.maximum_max_speed, shape=(2,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        lo = jnp.array((self.x_bounds[0], self.y_bounds[0], 0.0))
        hi = jnp.array((self.x_bounds[1], self.y_bounds[1], self.max_steps_in_ep))
        return spaces.Box(lo, hi, (3,), dtype=jnp.float64)


class DoubleGyreFlowCSDA(DoubleGyreFlowCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_speed

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
