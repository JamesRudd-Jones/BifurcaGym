"""
Wrappers exposing the pure JAX environments of HydroGym through the BifurcaGym API:
https://github.com/dynamicslab/hydrogym
https://proceedings.mlr.press/v283/lagemann25a.html

Requires the optional dependency group: `pdm install -G hydrogym`. HydroGym is imported lazily, so the rest of
BifurcaGym works without it. Only HydroGym's JAX backend is wrapped, its Firedrake, MAIA and NEK5000 backends cannot be
jitted or vmapped.

The HydroGym solvers are called directly rather than through their gymnax `step_env`, as in hydrogym 1.0.0 the
Kolmogorov `step_env` raises under gymnax>=1.0 (no `is_terminated`), and the Channel `step_env` computes an unused
gradient of the reward through the DNS each step. This relies on hydrogym internals, hence the `~=1.0.0` pin.
"""

import importlib
import jax
import jax.numpy as jnp
import numpy as np
from bifurcagym.envs import base_env, utils
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple
import chex
import cmocean


def _import_hydrogym(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError("HydroGym environments need the optional hydrogym dependencies, "
                          "install with `pdm install -G hydrogym`.") from e


# # # Kolmogorov Flow
########################################################################################################################
KOLMOGOROV_MAX_OBS = 6.0  # random control rollouts reach ~3.7 after 25 steps with TKE still slowly rising


@struct.dataclass
class KolmogorovEnvState(base_env.EnvState):
    omega_hat: jnp.ndarray  # Fourier vorticity at the end of the last action
    trajectory: jnp.ndarray  # Fourier vorticity saved during the last action, obs and reward are averaged over this


@struct.dataclass
class KolmogorovEnvParams:
    max_steps_in_ep: int = struct.field(False, default=1000)

    max_control: float = 0.5
    maximum_max_control: float = struct.field(False, default=0.5)  # maximum to ensure correct scaling

    init_noise: float = 0.1  # RMS of the random initial vorticity perturbation, 0 reproduces HydroGym's reset
    noise_kmax: int = struct.field(False, default=4)  # Only perturb wavenumbers with |k| <= noise_kmax


class KolmogorovFlowCSCA(base_env.BaseEnvironment):
    """
    2D Kolmogorov flow at Re=200 forced by four sinusoidal body forces (wavenumbers 4-7), observing the mean velocity
    magnitude on an 8x8 grid over each action. Reward is -(reward_alpha * mean TKE + sum|action|) as in HydroGym, set
    reward_alpha > 0 to penalise turbulent kinetic energy.

    HydroGym's reset always starts from the same field, here a low wavenumber vorticity perturbation seeded by the
    reset key is added before the uncontrolled burn in, so episodes differ.
    """

    def __init__(self,
                 dt: float = 1e-3,
                 action_time: float = 10.0,
                 reward_alpha: float = 0.0,
                 reynolds: float = 200.0,
                 **env_kwargs):
        super().__init__(**env_kwargs)

        kolmogorov = _import_hydrogym("hydrogym.jax.envs.kolmogorov")

        class _KolmogorovFlow(kolmogorov.KolmogorovFlow):
            # HydroGym's rollout and reward read the default params, not those passed in, so override them here
            @property
            def default_params(self):
                return super().default_params.replace(action_time=action_time, reward_alpha=reward_alpha)

        self.hg_env = _KolmogorovFlow(env_config={"dt": dt}, flow_config={"Re": reynolds})
        self.hg_params = self.hg_env.default_params
        self.n, self.m = self.hg_env.n, self.hg_env.m
        self.action_dim: int = self.hg_params.action_dim
        self.obs_dim: int = self.hg_params.obs_dim

        self.action_array: chex.Array = utils.action_permutations_generation(self.action_dim)

        self.requires_float64: bool = False

    @property
    def default_params(self) -> KolmogorovEnvParams:
        return KolmogorovEnvParams()

    def step_env(self,
                 input_action: chex.Numeric,
                 state: KolmogorovEnvState,
                 params: KolmogorovEnvParams,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, KolmogorovEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action, params)

        control_field = self.hg_env._control_field(action, self.hg_params)
        omega_hat, trajectory = self.hg_env._rollout(state.omega_hat, self.hg_params, control_field)

        new_state = KolmogorovEnvState(omega_hat=omega_hat, trajectory=trajectory, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def reset_env(self, params: KolmogorovEnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, KolmogorovEnvState]:
        omega_hat_0 = self.hg_env.flow.initialize_state()

        kx = jnp.fft.fftfreq(self.n) * self.n
        ky = jnp.fft.rfftfreq(self.m) * self.m
        KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
        low_k_mask = (KX ** 2 + KY ** 2) <= params.noise_kmax ** 2

        noise = jnp.fft.irfftn(jnp.fft.rfftn(jax.random.normal(key, (self.n, self.m))) * low_k_mask, s=(self.n, self.m))
        noise = params.init_noise * noise / jnp.std(noise)
        omega_hat_0 = omega_hat_0 + jnp.fft.rfftn(noise).astype(omega_hat_0.dtype)

        # Uncontrolled burn in for one action length, as in HydroGym's reset
        omega_hat, trajectory = self.hg_env._rollout(omega_hat_0, self.hg_params, None)

        state = KolmogorovEnvState(omega_hat=omega_hat, trajectory=trajectory, time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: KolmogorovEnvState,
                                 state_tp1: KolmogorovEnvState,
                                 params: KolmogorovEnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        action = self.action_convert(input_action_t, params)
        reward = self.hg_env._reward(action, state_tp1.trajectory, self.hg_params)

        done = (state_tp1.time >= params.max_steps_in_ep) | jnp.any(jnp.isnan(state_tp1.omega_hat))

        return reward, done

    def action_convert(self, action: chex.Numeric, params: KolmogorovEnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control)

    def get_obs(self, state: KolmogorovEnvState, key: chex.PRNGKey = None) -> chex.Array:
        return self.hg_env._trajectory_mean_obs(state.trajectory)

    def get_state(self, obs: chex.Array, params: KolmogorovEnvParams) -> KolmogorovEnvState:
        raise ValueError("Cannot reconstruct full state: This environment is partially observable.")

    def render_traj(self, trajectory_state: KolmogorovEnvState, params: KolmogorovEnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        vorticity = np.fft.irfftn(np.asarray(trajectory_state.omega_hat), s=(self.n, self.m), axes=(-2, -1))
        v_max = float(np.abs(vorticity).max())

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        im = ax.imshow(vorticity[0].T, cmap=cmocean.cm.curl, origin="lower", vmin=-v_max, vmax=v_max)
        fig.colorbar(im, ax=ax, shrink=0.5, label="Vorticity")

        def update(frame):
            im.set_array(vorticity[frame].T)
            return im,

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(vorticity),
                                       interval=200,
                                       blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "HydroGym-Kolmogorov-v0"

    def action_space(self, params: KolmogorovEnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, (self.action_dim,), dtype=jnp.float64)

    def observation_space(self, params: KolmogorovEnvParams) -> spaces.Box:
        # Mean velocity magnitudes, bound found empirically from uncontrolled and random control rollouts
        return spaces.Box(0.0, KOLMOGOROV_MAX_OBS, (self.obs_dim,), dtype=jnp.float64)


class KolmogorovFlowCSDA(KolmogorovFlowCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: KolmogorovEnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: KolmogorovEnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))


# # # Channel Flow
########################################################################################################################
CHANNEL_OBS_LOW = -5.0  # random control rollouts give streamwise U in ~[-1, 9.7] and spanwise W in ~[-1.8, 1.8]
CHANNEL_OBS_HIGH = 15.0


@struct.dataclass
class ChannelEnvState(base_env.EnvState):
    U: jnp.ndarray
    V: jnp.ndarray
    W: jnp.ndarray


@struct.dataclass
class ChannelEnvParams:
    max_steps_in_ep: int = struct.field(False, default=100)

    max_control: float = 1.0
    maximum_max_control: float = struct.field(False, default=1.0)  # maximum to ensure correct scaling


class ChannelFlowCSCA(base_env.BaseEnvironment):
    """
    3D turbulent channel flow at Re_tau=180 (72^3 pseudo-spectral DNS) with 24 wall actuators, observing streamwise
    and spanwise velocity on an 8x8 grid in a near wall plane. Reward is minus the mean wall shear stress.

    The initial turbulent field is downloaded from HydroGym's HuggingFace dataset on first use (cached in
    ~/.cache/hydrogym), pass `initial_field_dir` to load it from disk instead. Reset is deterministic as in HydroGym, as
    perturbing the velocity field directly would break incompressibility. Importing this HydroGym module enables
    jax_enable_x64 globally.
    """

    def __init__(self,
                 dns_steps_per_env_step: int = 50,
                 initial_field_dir: str = None,
                 **env_kwargs):
        super().__init__(**env_kwargs)

        self.channel = _import_hydrogym("hydrogym.jax.envs.channel")

        class _ChannelFlow(self.channel.ChannelFlowSpectralEnv):
            # HydroGym's DNS rollout reads nsteps from the default params, not those passed in, so override them here
            @property
            def default_params(self):
                return super().default_params.replace(nsteps=dns_steps_per_env_step)

        env_config = {} if initial_field_dir is None else {"initial_field_dir": initial_field_dir}
        self.hg_env = _ChannelFlow(env_config)
        self.hg_params = self.hg_env.default_params
        self.action_dim: int = self.hg_params.action_dim
        self.obs_dim: int = self.hg_params.obs_subsample ** 2 * self.hg_params.obs_include_components

        self.requires_float64: bool = False

    @property
    def default_params(self) -> ChannelEnvParams:
        return ChannelEnvParams()

    def step_env(self,
                 input_action: chex.Numeric,
                 state: ChannelEnvState,
                 params: ChannelEnvParams,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, ChannelEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action, params)

        _, (U, V, W) = self.hg_env._wss_with_aux(self._hg_state(state), action, self.hg_params)

        new_state = ChannelEnvState(U=U, V=V, W=W, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def _hg_state(self, state: ChannelEnvState):
        # HydroGym counts time in DNS steps, which sets the phase of its time dependent forcing
        return self.channel.ChannelEnvState(time=state.time * self.hg_params.nsteps,
                                            U=state.U,
                                            V=state.V,
                                            W=state.W,
                                            dt=jnp.asarray(self.hg_params.dt),
                                            terminal=jnp.bool_(False))

    def reset_env(self, params: ChannelEnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, ChannelEnvState]:
        state = ChannelEnvState(U=self.hg_env.U0, V=self.hg_env.V0, W=self.hg_env.W0, time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: ChannelEnvState,
                                 state_tp1: ChannelEnvState,
                                 params: ChannelEnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        reward = -self.channel.wss_compute(4, state_tp1.U, z=self.hg_env.equation.z)

        done = (state_tp1.time >= params.max_steps_in_ep) | jnp.any(jnp.isnan(state_tp1.U))

        return reward, done

    def action_convert(self, action: chex.Numeric, params: ChannelEnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control)

    def get_obs(self, state: ChannelEnvState, key: chex.PRNGKey = None) -> chex.Array:
        return self.channel.get_obs_spectral_channel(state, self.hg_params, self.hg_env.Xi_obs, self.hg_env.Yi_obs)

    def get_state(self, obs: chex.Array, params: ChannelEnvParams) -> ChannelEnvState:
        raise ValueError("Cannot reconstruct full state: This environment is partially observable.")

    def render_traj(self, trajectory_state: ChannelEnvState, params: ChannelEnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        # Streamwise velocity in the near wall plane that is observed
        us = np.asarray(trajectory_state.U[..., self.hg_params.k_det])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        im = ax.imshow(us[0].T, cmap=cmocean.cm.speed, origin="lower", vmin=float(us.min()), vmax=float(us.max()))
        fig.colorbar(im, ax=ax, shrink=0.5, label="Streamwise Velocity")

        def update(frame):
            im.set_array(us[frame].T)
            return im,

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(us),
                                       interval=200,
                                       blit=True)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "HydroGym-Channel-v0"

    def action_space(self, params: ChannelEnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, (self.action_dim,), dtype=jnp.float64)

    def observation_space(self, params: ChannelEnvParams) -> spaces.Box:
        # Near wall velocity components, bound found empirically from uncontrolled and random control rollouts
        return spaces.Box(CHANNEL_OBS_LOW, CHANNEL_OBS_HIGH, (self.obs_dim,), dtype=jnp.float64)
