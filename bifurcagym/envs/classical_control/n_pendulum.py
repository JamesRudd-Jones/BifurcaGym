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
    theta: jnp.ndarray      # shape (n_links,)
    theta_dot: jnp.ndarray  # shape (n_links,)


@struct.dataclass
class EnvParams:
    action_array: jnp.ndarray = struct.field(False, default=(jnp.array((0.0, 1.0, -1.0))))
    dt: float = struct.field(False, default=0.05)
    horizon: int = struct.field(False, default=200)
    max_steps_in_ep: int = struct.field(False, default=1000)
    periodic_dim: jnp.ndarray = struct.field(False, default=jnp.array((1, 0)))  # TODO is this the best way?

    n_links: int = struct.field(False, default=2)

    # for now links are standardised to be the same
    max_speed: float = 8.0
    maximum_max_speed: float = struct.field(False, default=8.0)  # maximum to ensure correct scaling
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0

    max_torque: float = 2.0
    maximum_max_torque: float = struct.field(False, default=2.0)  # maximum to ensure correct scaling


class NPendulumCSDA(base_env.BaseEnvironment):
    """
    N-link pendulum environment where control torque is applied only at the base joint (index 0)
    Angles are relative joint angles; absolute link orientation = cumsum(theta)
    Approximate dynamics: gravity acts on each link (via absolute orientation), torques only at base joint; inertial coupling is not modelled here
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

        angle_abs = jnp.cumsum(state.theta)  # abs angles for each link: angle_abs[i] = sum(theta[:i+1])

        theta_ddot = (-params.gravity / params.length) * jnp.sin(angle_abs)
        # add torque term to the base joint acceleration only (divide by moment-like term)
        torque_inertia_term = params.mass * (params.length ** 2)
        # create torque acceleration contribution: shape (n_links,)
        torque_acc = jnp.zeros((params.n_links,))
        torque_acc = torque_acc.at[0].set(action / torque_inertia_term)

        theta_ddot = theta_ddot + torque_acc

        new_theta_dot = state.theta_dot + theta_ddot * params.dt
        new_theta_dot = jnp.clip(new_theta_dot, -params.max_speed, params.max_speed)

        new_theta = state.theta + new_theta_dot * params.dt
        new_theta = self._angle_normalise(new_theta)

        new_state = EnvState(theta=new_theta, theta_dot=new_theta_dot, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    @staticmethod
    def _angle_normalise(x: jnp.ndarray) -> jnp.ndarray:
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        high_theta = jnp.pi * 0.5
        high_thdot = 1.0
        key1, key2 = jrandom.split(key)
        theta_init = jrandom.uniform(key1, shape=(params.n_links,), minval=-high_theta, maxval=high_theta)
        thdot_init = jrandom.uniform(key2, shape=(params.n_links,), minval=-high_thdot, maxval=high_thdot)

        state = EnvState(theta=theta_init,
                         theta_dot=thdot_init,
                         time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        # penalise absolute link angles (keep chain pointing down), velocities, and torque at base only
        angle_abs_tp1 = jnp.cumsum(state_tp1.theta)
        cost_angles = jnp.sum(self._angle_normalise(angle_abs_tp1) ** 2)
        cost_vel = 0.1 * jnp.sum(state_tp1.theta_dot ** 2)
        action_vec = self.action_convert(input_action_t, params)
        cost_action = 0.001 * jnp.sum(action_vec ** 2)

        done = jnp.array(state_tp1.time >= params.max_steps_in_ep)  # TODO state_t or state_tp1

        return -(cost_angles + cost_vel + cost_action), done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Array:
        return params.action_array[action.squeeze()] * params.max_torque

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.concatenate([state.theta, state.theta_dot])

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        n = params.n_links
        return EnvState(theta=obs[:n], theta_dot=obs[n:], time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        # compute link endpoints for a trajectory: theta trajectory shape (T, n_links)
        theta_traj = np.asarray(trajectory_state.theta)      # can be (T, n) or (n,) single time
        lengths = np.asarray(params.length)

        # Ensure theta_traj is (T, n)
        if theta_traj.ndim == 1:
            theta_traj = theta_traj[np.newaxis, ...]
        T = theta_traj.shape[0]

        def endpoints_for_frame(theta_frame, length):
            # theta_frame: (n,)
            abs_angles = np.cumsum(theta_frame)  # absolute orientation of each link
            xs = []
            ys = []
            x, y = 0.0, 0.0
            for ang in abs_angles:
                dx = -length * np.sin(ang)
                dy = length * np.cos(ang)
                x = x + dx
                y = y + dy
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        max_length = np.max(lengths)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        reach = max_length * params.n_links
        ax.set_xlim(-reach * 1.2, reach * 1.2)
        ax.set_ylim(-reach * 1.2, reach * 1.2)
        ax.set_aspect('equal')
        ax.grid(True)

        # initial coordinates
        xs0, ys0 = endpoints_for_frame(theta_traj[0], lengths[0])
        line, = ax.plot([0.0] + xs0.tolist(), [0.0] + ys0.tolist(), lw=3, c='k', marker='o')
        circle_patches = []
        bob_radius = 0.08
        for (x_i, y_i) in zip(xs0, ys0):
            circle = plt.Circle((float(x_i), float(y_i)), bob_radius, fc='r', zorder=3)
            ax.add_patch(circle)
            circle_patches.append(circle)

        def update(frame):
            xs, ys = endpoints_for_frame(theta_traj[frame], lengths[frame])
            line.set_data([0.0] + xs.tolist(), [0.0] + ys.tolist())
            for (circle, x_i, y_i) in zip(circle_patches, xs, ys):
                circle.set_center((float(x_i), float(y_i)))
            return (line, *circle_patches)

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=T,
                                       interval=params.dt * 1000,
                                       blit=True
                                       )
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()
    # TODO sort out render_traj as unsure it actually works as of now

    @property
    def name(self) -> str:
        return "NPendulum-v0"

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(len(params.action_array))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        high = jnp.concatenate([jnp.ones((params.n_links,)) * jnp.pi, jnp.ones((params.n_links,)) * params.maximum_max_speed])
        return spaces.Box(-high, high, (2 * params.n_links,), dtype=jnp.float32)

    def reward_space(self, params: EnvParams) -> spaces.Box:
        # TODO actually add in the reward space bounds as unsure at the moment
        return spaces.Box(-100.0, 0.0, (()), dtype=jnp.float32)


class NPendulumCSCA(NPendulumCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return jnp.clip(action, -params.max_torque, params.max_torque).squeeze()

    def action_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-params.maximum_max_torque, params.maximum_max_torque, shape=(1,))
