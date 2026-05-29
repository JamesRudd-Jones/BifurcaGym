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
    U0: float = 0.6266
    L: float = 1.0

    k: chex.Array = jnp.array([2.0 * 1.0, 2.0 * 2.0, 2.0 * 3.0]) / 6.371
    epsilon: chex.Array = jnp.array([0.0075, 0.15, 0.3])

    max_speed: float = 0.1
    maximum_max_speed: float = struct.field(False, default=0.1)  # maximum to ensure correct scaling

    @property
    def c(self) -> chex.Array:
        return jnp.array([0.1446, 0.205, 0.461]) * self.U0


class BickleyJetFlowCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = 0.1
        self.horizon: int = 25
        self.max_steps_in_ep: int = 500

        self.action_array: chex.Array = jnp.array(((0.0, 0.0),
                                                   (-1.0, 0.0),
                                                   (-1.0, -1.0),
                                                   (0.0, -1.0),
                                                   (1.0, 0.0),
                                                   (1.0, 1.0),
                                                   (0.0, 1.0),
                                                   ))

        self.requires_float64: bool = True

        self.x_bounds: chex.Array = jnp.array((0.0, 4.0 * jnp.pi))
        self.y_bounds: chex.Array = jnp.array((-3.0, 3.0))
        self.goal_state: chex.Array = jnp.array((6.0, -2.0))
        self.x_len: chex.Array = self.x_bounds[1] - self.x_bounds[0]

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

        # Handle Boundaries:
        new_x = new_x % self.x_len  # X is Periodic
        new_y = jnp.clip(new_y, self.y_bounds[0], self.y_bounds[1])

        new_state = EnvState(x=new_x, y=new_y, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def _get_flow_velocity(self, x: chex.Array, y: chex.Array, t: float, params: EnvParams) -> Tuple[chex.Array, chex.Array]:
        # Background zonal flow: U = U0 * sech^2(y/L)
        # Streamfunction psi_0 = -U0 * L * tanh(y/L)

        # Calculate perturbations from the 3 waves
        # sigma = k * c (phase speed relation)
        sigma = params.k * params.c

        # Reshape for broadcasting if inputs are arrays
        k_grid = params.k
        sigma_grid = sigma
        eps_grid = params.epsilon

        # Argument for cosine term: kx - sigma*t
        # Note: We must be careful with shapes if x/y are batches.
        # Assuming scalar or simple array x/y here.

        phase = jnp.outer(x, k_grid) - sigma_grid * t
        # If x is scalar, phase is shape (3,). If x is (N,), phase is (N, 3)
        # We adjust dimensions to ensure summation works correctly.
        phase = phase.squeeze()

        # Perturbation terms
        # u_n = 2 * U0 * eps_n * sech^2(y) * tanh(y) * cos(k_n x - sigma_n t)
        # v_n = -U0 * L * eps_n * k_n * sech^2(y) * sin(k_n x - sigma_n t)

        sech_sq = (1.0 / jnp.cosh(y / params.L)) ** 2
        tanh_y = jnp.tanh(y / params.L)

        # Summation over the 3 waves
        cos_phase = jnp.cos(phase)
        sin_phase = jnp.sin(phase)

        u_perturb = jnp.sum(2 * params.U0 * eps_grid * sech_sq * tanh_y * cos_phase)
        v_perturb = jnp.sum(-params.U0 * params.L * eps_grid * k_grid * sech_sq * sin_phase)

        u_bg = params.U0 * sech_sq

        return u_bg + u_perturb, v_perturb

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, _key = jrandom.split(key)

        start_x = jrandom.uniform(key, minval=0.5, maxval=self.x_bounds[1] - 0.5)
        start_y = jrandom.uniform(_key, minval=1.5, maxval=2.5)

        state = EnvState(x=start_x, y=start_y, time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        # Standard distance metric with periodic adjustment for X?
        # For simplicity assume standard Euclidean, but strict periodicity usually requires dx = min(|x1-x2|, L - |x1-x2|).
        dx = jnp.abs(state_t.x - self.goal_state[0])
        dx = jnp.minimum(dx, self.x_len - dx)  # Periodic distance
        dy = state_t.y - self.goal_state[1]

        dist_to_target = jnp.sqrt(dx ** 2 + dy ** 2)

        done = dist_to_target < 0.2

        time_up = state_tp1.time >= self.max_steps_in_ep

        done = jnp.logical_or(done, time_up)

        reward = jnp.array(-0.01)  # Time penalty
        # TODO a goal based reward or some distance metric?

        reward_adder = jax.lax.select(done, 10.0, 0.0)
        reward += reward_adder

        return reward, done

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
        X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)

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

        cmap_val = cmocean.cm.deep

        get_flow_vel = jax.vmap(jax.vmap(self._get_flow_velocity, in_axes=(0, 0, None, None)), in_axes=(0, 0, None, None))

        def get_curr_param(params, time_idx):
            return jax.tree.map(lambda x: x[time_idx] if getattr(x, 'ndim', 0) > 0 else x, params)

        initial_U_coarse, initial_V_coarse = get_flow_vel(X_coarse, Y_coarse, times[0] * self.dt, get_curr_param(params, 0))
        initial_U_fine, initial_V_fine = get_flow_vel(X_fine, Y_fine, times[0] * self.dt, get_curr_param(params, 0))
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

            U_coarse, V_coarse = get_flow_vel(X_coarse, Y_coarse, times[frame] * self.dt, get_curr_param(params, frame))
            U_fine, V_fine = get_flow_vel(X_fine, Y_fine, times[frame] * self.dt, get_curr_param(params, frame))
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
        return "BickleyJetFlow-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_speed, params.maximum_max_speed, shape=(2,), dtype=jnp.float64)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        low = jnp.array([self.x_bounds[0], self.y_bounds[0], 0.0], dtype=jnp.float64)
        high = jnp.array([self.x_bounds[1], self.y_bounds[1], self.max_steps_in_ep], dtype=jnp.float64)
        return spaces.Box(low, high, shape=(3,), dtype=jnp.float64)


class BickleyJetFlowCSDA(BickleyJetFlowCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_speed

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
