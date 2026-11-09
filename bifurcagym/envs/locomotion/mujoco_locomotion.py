"""
Hopper and Walker2d from Gymnasium's MuJoCo suite (v5), simulated with MJX, the JAX implementation of MuJoCo:
https://gymnasium.farama.org/environments/mujoco/
https://mujoco.readthedocs.io/en/stable/mjx.html

Requires the optional dependency group: `pdm install -G mujoco`. MuJoCo is imported lazily, so the rest of BifurcaGym
works without it. The models in `assets/` are Gymnasium's `hopper.xml` and `walker2d_v5.xml`, unchanged (MIT licence,
see `assets/LICENSE_GYMNASIUM`).

Rewards, termination, frame skip, reset noise and observations follow the Gymnasium v5 defaults, except that:
    - The torso pitch in the observation is wrapped to [-pi, pi] (a periodic dimension), so the observation space is
      bounded for the normalised wrapper. This only differs from Gymnasium when terminate_when_unhealthy=False.
    - The constraint solver warm start is not carried between steps, so the state is just (qpos, qvel). Each step is
      solved to MuJoCo's solver tolerance either way.

Unactuated, or held upright by a joint controller, these robots are dissipative and settle (all Lyapunov exponents
<= 0). Under sustained actuation the intermittent ground contacts make them chaotic, with leading exponents of a few
per second, see tests/physics/test_locomotion.py.
"""

import contextlib
import importlib
import io
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from bifurcagym.envs import base_env, utils
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple
import chex


ASSET_DIR = Path(__file__).parent / "assets"


def _import_mujoco():
    try:
        mujoco = importlib.import_module("mujoco")
        # MJX prints a notice when the optional Warp backend is missing, which is not used here
        with contextlib.redirect_stdout(io.StringIO()):
            mjx = importlib.import_module("mujoco.mjx")
    except ImportError as e:
        raise ImportError("MuJoCo environments need the optional mujoco dependencies, "
                          "install with `pdm install -G mujoco`.") from e
    return mujoco, mjx


@struct.dataclass
class EnvState(base_env.EnvState):
    qpos: jnp.ndarray  # (rootx, rootz, rooty, joint angles)
    qvel: jnp.ndarray


@struct.dataclass
class EnvParams:
    max_steps_in_ep: int = struct.field(False, default=1000)

    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0
    terminate_when_unhealthy: bool = True

    # Healthy if healthy_z_low < z < healthy_z_high, |pitch| < healthy_angle and |qpos[2:], qvel| < healthy_state_max
    healthy_z_low: float = 0.7
    healthy_z_high: float = float("inf")
    healthy_angle: float = 0.2
    healthy_state_max: float = 100.0

    reset_noise_scale: float = 5e-3

    max_control: float = 1.0
    maximum_max_control: float = struct.field(False, default=1.0)  # maximum to ensure correct scaling


class HopperCSCA(base_env.BaseEnvironment):
    """
    Planar one legged hopper (3 actuated joints), rewarded for forward velocity while staying upright. Observes
    (z, pitch, joint angles, clipped velocities), the forward position is excluded as in Gymnasium.
    """

    xml_file: str = "hopper.xml"

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.mujoco, self.mjx = _import_mujoco()
        self.mj_model = self.mujoco.MjModel.from_xml_path(str(ASSET_DIR / self.xml_file))
        self.mjx_model = self.mjx.put_model(self.mj_model)
        self.mjx_data = self.mjx.make_data(self.mjx_model)

        self.frame_skip: int = 4
        self.dt: float = self.mj_model.opt.timestep * self.frame_skip
        self.nq: int = self.mj_model.nq
        self.nv: int = self.mj_model.nv
        self.action_dim: int = self.mj_model.nu
        self.max_obs_velocity: float = 10.0
        self.init_qpos: chex.Array = jnp.asarray(self.mj_model.qpos0)

        # Observation bounds: z, pitch, joint ranges (with slack as MuJoCo joint limits are soft), clipped velocities
        joint_range = self.mj_model.jnt_range[3:]
        self.obs_low: chex.Array = jnp.concatenate((jnp.array((-0.5, -jnp.pi)),
                                                    joint_range[:, 0] - 0.2,
                                                    -self.max_obs_velocity * jnp.ones(self.nv)))
        self.obs_high: chex.Array = jnp.concatenate((jnp.array((3.0, jnp.pi)),
                                                     joint_range[:, 1] + 0.2,
                                                     self.max_obs_velocity * jnp.ones(self.nv)))
        self.obs_dim: int = self.obs_low.shape[0]

        self.periodic_dim: chex.Array = jnp.zeros(self.obs_dim).at[1].set(1)

        self.action_array: chex.Array = utils.action_permutations_generation(self.action_dim)

        self.requires_float64: bool = False

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

        qpos, qvel = self._physics_step(state.qpos, state.qvel, action)

        new_state = EnvState(qpos=qpos, qvel=qvel, time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    def _physics_step(self, qpos: chex.Array, qvel: chex.Array, ctrl: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Holds the control for frame_skip MuJoCo timesteps, as Gymnasium's do_simulation"""
        data = self.mjx_data.replace(qpos=qpos.astype(self.mjx_data.qpos.dtype),
                                     qvel=qvel.astype(self.mjx_data.qvel.dtype),
                                     ctrl=ctrl.astype(self.mjx_data.ctrl.dtype))
        data = jax.lax.fori_loop(0, self.frame_skip, lambda _, d: self.mjx.step(self.mjx_model, d), data)
        return data.qpos, data.qvel

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        pos_key, vel_key = jrandom.split(key)
        noise = params.reset_noise_scale
        qpos = self.init_qpos + jrandom.uniform(pos_key, (self.nq,), minval=-noise, maxval=noise)
        qvel = jrandom.uniform(vel_key, (self.nv,), minval=-noise, maxval=noise)
        state = EnvState(qpos=qpos.astype(self.mjx_data.qpos.dtype), qvel=qvel.astype(self.mjx_data.qvel.dtype), time=0)

        return self.get_obs(state), state

    def is_healthy(self, state: EnvState, params: EnvParams) -> chex.Array:
        z, angle = state.qpos[1], state.qpos[2]
        rest = jnp.concatenate((state.qpos[2:], state.qvel))
        return ((params.healthy_z_low < z) & (z < params.healthy_z_high) &
                (jnp.abs(angle) < params.healthy_angle) &
                jnp.all(jnp.abs(rest) < params.healthy_state_max))

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        action_t = self.action_convert(input_action_t, params)
        healthy = self.is_healthy(state_tp1, params)

        x_velocity = (state_tp1.qpos[0] - state_t.qpos[0]) / self.dt
        reward = (params.forward_reward_weight * x_velocity
                  + params.healthy_reward * healthy
                  - params.ctrl_cost_weight * jnp.sum(action_t ** 2))

        done = ((state_tp1.time >= params.max_steps_in_ep)
                | (params.terminate_when_unhealthy & ~healthy)
                | jnp.any(jnp.isnan(state_tp1.qpos)))

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_control, params.max_control)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        pitch = ((state.qpos[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        return jnp.concatenate((state.qpos[1:2],
                                pitch[None],
                                state.qpos[3:],
                                jnp.clip(state.qvel, -self.max_obs_velocity, self.max_obs_velocity)))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        raise ValueError("Cannot reconstruct full state: the forward position is not observed and velocities are clipped.")

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        mj_data = self.mujoco.MjData(self.mj_model)
        capsules = [g for g in range(self.mj_model.ngeom) if self.mj_model.geom_type[g] == self.mujoco.mjtGeom.mjGEOM_CAPSULE]

        def segments(qpos):
            # Capsule end points in the x-z plane from the forward kinematics
            mj_data.qpos[:] = qpos
            self.mujoco.mj_kinematics(self.mj_model, mj_data)
            ends = []
            for g in capsules:
                centre, axis = mj_data.geom_xpos[g], mj_data.geom_xmat[g].reshape(3, 3)[:, 2]
                half = self.mj_model.geom_size[g, 1]
                ends.append(((centre - half * axis)[[0, 2]], (centre + half * axis)[[0, 2]]))
            return ends

        qposs = np.asarray(trajectory_state.qpos)
        frames = [segments(q) for q in qposs]
        width = 2.0

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(self.name)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        ax.set_ylim(-0.1, 2.0)
        floor, = ax.plot([], [], c="k", lw=1)
        lines = [ax.plot([], [], lw=6, solid_capstyle="round", c=f"C{i % 10}")[0] for i in range(len(capsules))]

        def update(frame):
            x_c = qposs[frame, 0]
            ax.set_xlim(x_c - width, x_c + width)
            floor.set_data([x_c - width, x_c + width], [0.0, 0.0])
            for line, (a, b) in zip(lines, frames[frame]):
                line.set_data([a[0], b[0]], [a[1], b[1]])
            return lines + [floor]

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(frames),
                                       interval=self.dt * 1000,
                                       blit=False)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "Hopper-v0"

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_control, params.maximum_max_control, (self.action_dim,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(self.obs_low, self.obs_high, (self.obs_dim,))


class HopperCSDA(HopperCSCA):
    """Bang bang control, each joint torque in {0, +max_control, -max_control}"""
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))


class Walker2dCSCA(HopperCSCA):
    """
    Planar biped (6 actuated joints), rewarded for forward velocity while staying upright. Observes (z, pitch, joint
    angles, clipped velocities), the forward position is excluded as in Gymnasium.
    """

    xml_file: str = "walker2d_v5.xml"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(healthy_z_low=0.8, healthy_z_high=2.0, healthy_angle=1.0, healthy_state_max=float("inf"))

    @property
    def name(self) -> str:
        return "Walker2d-v0"


class Walker2dCSDA(Walker2dCSCA):
    """Bang bang control, each joint torque in {0, +max_control, -max_control}"""
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()] * params.max_control

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))
