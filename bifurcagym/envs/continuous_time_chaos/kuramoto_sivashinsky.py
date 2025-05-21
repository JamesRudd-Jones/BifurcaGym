import jax.numpy as jnp
import numpy as np
from bifurcagym.envs import base_env
from gymnax.environments import spaces
from flax import struct
from typing import Any, Dict, Optional, Tuple, Union
import chex
from typing import Optional
import jax


# jax.config.update("jax_enable_x64", True)  # TODO unsure if need or not but will check results


@struct.dataclass
class EnvState(base_env.EnvState):
    u: jnp.ndarray
    time: int


class KuramotoSivashinskyCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.x: jnp.ndarray = jnp.array(np.loadtxt("../bifurcagym/envs/continuous_time_chaos/ks_files/x.dat"))  # select space discretization of the target solution
        self.U_bf: jnp.ndarray = jnp.array(np.loadtxt('../bifurcagym/envs/continuous_time_chaos/ks_files/u2.dat'))  # select u1, u2 or u3 as target solution

        self.max_control: float = 0.1
        self.horizon: int = 200

        self.state_dim: int = 8
        self.action_dim: int = 4

        N = self.x.size
        self.dt: float = 0.05
        self.L: int = 22
        self.x_S = jnp.arange(N) * self.L / N
        k_K = N * jnp.fft.fftfreq(N)[0:N // 2 + 1] * 2 * jnp.pi / self.L
        self.ik_K = 1j * k_K  # spectral derivative operator
        self.lin_K = k_K ** 2 - k_K ** 4  # Fourier multipliers for linear term
        self.a_dim = self.action_dim
        self.s_dim = self.state_dim

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

    def nlterm(self, u, f):
        # compute tendency from nonlinear term. advection + forcing
        ur = jnp.fft.irfft(u, axis=-1)
        return -0.5 * self.ik_K * jnp.fft.rfft(ur ** 2, axis=-1) + f

    def step_env(self,
                 input_action: Union[int, float, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # forcing shape
        dum_SA = self.B_SA * input_action.T  # TODO check this transpose
        f0_S = jnp.sum(dum_SA, axis=-1)

        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        u_K = jnp.fft.rfft(state.u, axis=-1)
        f_K = jnp.fft.rfft(f0_S, axis=-1)
        u_save_K = u_K.copy()  # TODO is this required?

        def _runge_kutta_update(runner, unused):
            u_K, ind = runner
            dt = self.dt / (3 - ind)
            u_K = u_save_K + dt * self.nlterm(u_K, f_K)
            u_K = (u_K + 0.5 * self.lin_K * dt * u_save_K) / (1. - 0.5 * self.lin_K * dt)

            ind += 1

            return (u_K, ind), None

        final_runner_state = jax.lax.scan(_runge_kutta_update, (u_K, 0), None, 3)
        u_S = jnp.fft.irfft(final_runner_state[0][0], axis=-1)

        new_state = EnvState(u=u_S,
                         time=state.time + 1)

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {})

    def generative_step_env(self,
                            action: Union[int, float, chex.Array],
                            obs: chex.Array,
                            key: chex.PRNGKey,
                            ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        state = EnvState(u=obs, time=0)
        return self.step(action, state, key)

    def reward_function(self,
                    input_action_t: Union[int, float, chex.Array],
                    state_t: EnvState,
                    state_tp1: EnvState,
                    key: chex.PRNGKey,
                    ) -> chex.Array:
        reward = -jnp.linalg.norm(state_tp1.u - self.U_bf)
        return reward

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        u_S = jnp.array(np.loadtxt('../bifurcagym/envs/continuous_time_chaos/ks_files/u3.dat'))
        state = EnvState(u=u_S,
                         time=0)  # TODO is this okay?
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, key=None):
        return state.u[5::self.x_S.shape[0] // self.s_dim]

    def is_done(self, state: EnvState):
        return jnp.array(False)

    @property
    def name(self) -> str:
        return "KS-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_control, self.max_control, (self.action_dim,), dtype=jnp.float64)

    def observation_space(self) -> spaces.Box:
        high = 10  # TODO unsure of actual size should check
        return spaces.Box(-high, high, (self.state_dim,), dtype=jnp.float64)
