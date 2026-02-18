"""
An env ported to Jax based off of the following work:
https://royalsocietypublishing.org/doi/full/10.1098/rspa.2019.0351
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex
from pathlib import Path


@struct.dataclass
class EnvState(base_env.EnvState):
    u: jnp.ndarray
    time: int


class KuramotoSivashinskyCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        here = Path(__file__).resolve().parent
        self.data_dir = here / "ks_files"

        self.x: jnp.ndarray = jnp.array(np.loadtxt(self.data_dir / "x.dat"))  # select space discretisation of the target solution
        self.U_bf: jnp.ndarray = jnp.array(np.loadtxt(self.data_dir / 'u2.dat'))  # select u1, u2 or u3 as target solution
        N = self.x.size

        self.reward_ball: float = 0.01

        self.state_dim: int = 8
        self.action_dim: int = 4
        self.L: int = 22
        self.x_S = jnp.arange(N) * self.L / N
        k_K = N * jnp.fft.fftfreq(N)[0:N // 2 + 1] * 2 * jnp.pi / self.L
        self.ik_K = 1j * k_K  # spectral derivative operator
        self.lin_K = k_K ** 2 - k_K ** 4  # Fourier multipliers for linear term

        self.max_control: float = 0.1

        self.dt: float = 0.05
        self.max_steps_in_episode: int = int(500 // self.dt)

        sig = 0.4
        x_zero_A = self.x_S[-1] / self.action_dim * jnp.arange(0, self.action_dim)
        gaus = 1 / (jnp.sqrt(2 * jnp.pi) * sig) * jnp.exp(-0.5 * ((self.x_S - self.x_S[self.x_S.size // 2]) / sig) ** 2)

        def process_single(gaus, x_zero, x_S_center, dx):
            shift = jnp.floor(x_zero - x_S_center) / dx
            col = jnp.roll(gaus, shift.astype(int))
            col = col / jnp.max(col)
            return jnp.roll(col, 5)

        self.B_SA = jax.vmap(process_single, in_axes=(None, 0, None, None))(gaus,
                                                                            x_zero_A,
                                                                            self.x_S[self.x_S.size // 2],
                                                                            self.x_S[1] - self.x_S[0],
                                                                            ).T

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        # forcing shape
        dum_SA = self.B_SA * action.T  # TODO check this transpose
        f0_S = jnp.sum(dum_SA, axis=-1)

        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        u_K = jnp.fft.rfft(state.u, axis=-1)
        f_K = jnp.fft.rfft(f0_S, axis=-1)
        u_save_K = u_K.copy()

        def _runge_kutta_update(runner, unused):
            u_K, ind = runner
            dt = self.dt / (3 - ind)
            u_K = u_save_K + dt * self.nlterm(u_K, f_K)
            u_K = (u_K + 0.5 * self.lin_K * dt * u_save_K) / (1. - 0.5 * self.lin_K * dt)

            ind += 1

            return (u_K, ind), None

        final_runner_state = jax.lax.scan(_runge_kutta_update, (u_K, 0), None, 3)
        u_S = jnp.fft.irfft(final_runner_state[0][0], axis=-1)

        new_state = EnvState(u=u_S, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def nlterm(self, u, f):
        # compute tendency from nonlinear term. advection + forcing
        ur = jnp.fft.irfft(u, axis=-1)
        return -0.5 * self.ik_K * jnp.fft.rfft(ur ** 2, axis=-1) + f

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        u_S = jnp.array(np.loadtxt(self.data_dir / 'u3.dat'))
        state = EnvState(u=u_S, time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> Tuple[chex.Array, chex.Array]:
        reward = -jnp.linalg.norm(state_tp1.u - self.U_bf)

        state_done = jax.lax.select(jnp.linalg.norm(state_tp1.u - self.U_bf) < self.reward_ball,
                                    jnp.array(True),
                                    jnp.array(False))
        time_done = jax.lax.select(state_tp1.time >= self.max_steps_in_episode,
                                   jnp.array(True),
                                   jnp.array(False))
        done = jnp.logical_or(state_done, time_done)

        return reward, done

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_control, self.max_control)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None):
        return state.u[5::self.x_S.shape[0] // self.state_dim]

    def get_state(self, obs: chex.Array, jey: chex.PRNGKey = None) -> EnvState:
        return EnvState(u=obs[0], time=-1)

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        ax.set_xlim(trajectory_state.time[0], trajectory_state.time[-1])
        ax.set_xlabel("Time")
        ax.set_ylabel("x")

        v_min = float(trajectory_state.u.min())
        v_max = float(trajectory_state.u.max())

        im = ax.imshow(np.zeros((trajectory_state.time[0], trajectory_state.time[-1])).T,
                       cmap='viridis',
                       origin='lower',
                       vmin=v_min,
                       vmax=v_max)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.5, label="U Value")

        def update(frame):
            ax.clear()

            im = ax.imshow(trajectory_state.u[:frame].T,
                           cmap='viridis',
                           origin='lower',
                           vmin=v_min,
                           vmax=v_max)
            # cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.75, label="U Value")
            # TODO adding the above in causes some crazy animation, it is fun though so I have left in for your experience

            # Set the limits again, or you can adjust them dynamically as needed
            ax.set_title(self.name)
            ax.set_xlim(trajectory_state.time[0], trajectory_state.time[-1])
            ax.set_xlabel("Time")
            ax.set_ylabel("x")

            return im,

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(trajectory_state.time),
                                       interval=self.dt * 1000,  # Convert dt to milliseconds
                                       blit=True
                                       )
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

        # # Plotting the heatmap
        # plt.figure(figsize=(12, 8))
        # plt.imshow(trajectory_state.u[0:2].T,  # Transpose for spatial dimension on y-axis
        #            aspect='auto',
        #            origin='lower',  # Ensure y-axis starts at the bottom
        #            cmap='viridis',  # Use 'viridis' colormap
        #            # extent=[time_points[0], time_points[-1], spatial_coords[0], spatial_coords[-1]]
        #            )
        #
        # plt.colorbar(label='u(x, t)')  # Add a colorbar with a label
        # plt.xlabel('Time (t)')  # Label x-axis as Time
        # plt.ylabel('Spatial Coordinate (x)')  # Label y-axis as Spatial Coordinate
        # plt.title('Spatio-temporal Evolution of u(x,t)')  # Give the plot a title
        # plt.show()

    @property
    def name(self) -> str:
        return "KS-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, (self.action_dim,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        high = 10  # TODO unsure of actual size should check
        return spaces.Box(-high, high, (self.state_dim,), dtype=jnp.float64)


class KuramotoSivashinskyCSDA(KuramotoSivashinskyCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: jnp.ndarray = jnp.array((0.0, 1.0, -1.0))

        idx = jnp.arange(self.action_array.shape[0] ** self.action_dim)
        powers = self.action_array.shape[0] ** jnp.arange(self.action_dim)
        digits = (idx[:, None] // powers[None, :]) % self.action_array.shape[0]
        self.action_perms: jnp.ndarray = self.action_array[digits]
        # TODO this does not scale very nicely

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_perms[action.squeeze()] * self.max_control

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_perms))
