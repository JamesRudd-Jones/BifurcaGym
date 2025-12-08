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
    time: int


class NPendulumCSDA(base_env.BaseEnvironment):
    """
    N-link pendulum environment where control torque is applied only at the base joint (index 0)
    Angles are relative joint angles; absolute link orientation = cumsum(theta)
    Approximate dynamics: gravity acts on each link (via absolute orientation), torques only at base joint; inertial coupling is not modelled here
    """

    def __init__(self, n_links: int = 2, **env_kwargs):
        super().__init__(**env_kwargs)

        self.n_links = int(n_links)

        self.periodic_dim: chex.Array = jnp.array((1, 0))

        # for now links are standardised to be the same
        self.max_speed: float = 8.0
        self.gravity: float = 10.0
        self.mass: float = 1.0
        self.length: float = 1.0

        self.action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))
        self.max_torque: float = 2.0

        self.max_steps_in_episode: int = 1000

        self.horizon: int = 200
        self.dt: float = 0.05

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        angle_abs = jnp.cumsum(state.theta)  # abs angles for each link: angle_abs[i] = sum(theta[:i+1])

        theta_ddot = (-self.gravity / self.length) * jnp.sin(angle_abs)
        # add torque term to the base joint acceleration only (divide by moment-like term)
        torque_inertia_term = self.mass * (self.length ** 2)
        # create torque acceleration contribution: shape (n_links,)
        torque_acc = jnp.zeros((self.n_links,))
        torque_acc = torque_acc.at[0].set(action[0] / torque_inertia_term)

        theta_ddot = theta_ddot + torque_acc

        new_theta_dot = state.theta_dot + theta_ddot * self.dt
        new_theta_dot = jnp.clip(new_theta_dot, -self.max_speed, self.max_speed)

        new_theta = state.theta + new_theta_dot * self.dt
        new_theta = self._angle_normalise(new_theta)

        new_state = EnvState(theta=new_theta, theta_dot=new_theta_dot, time=state.time+1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                {})

    @staticmethod
    def _angle_normalise(x: jnp.ndarray) -> jnp.ndarray:
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        high_theta = jnp.pi * 0.5
        high_thdot = 1.0
        key1, key2 = jrandom.split(key)
        theta_init = jrandom.uniform(key1, shape=(self.n_links,), minval=-high_theta, maxval=high_theta)
        thdot_init = jrandom.uniform(key2, shape=(self.n_links,), minval=-high_thdot, maxval=high_thdot)

        state = EnvState(theta=theta_init,
                         theta_dot=thdot_init,
                         time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> Tuple[chex.Array, chex.Array]:
        # penalise absolute link angles (keep chain pointing down), velocities, and torque at base only
        angle_abs_tp1 = jnp.cumsum(state_tp1.theta)
        cost_angles = jnp.sum(self._angle_normalise(angle_abs_tp1) ** 2)
        cost_vel = 0.1 * jnp.sum(state_tp1.theta_dot ** 2)
        action_vec = self.action_convert(input_action_t)
        cost_action = 0.001 * jnp.sum(action_vec ** 2)

        done = jnp.array(state_tp1.time >= self.max_steps_in_episode)  # TODO state_t or state_tp1

        return -(cost_angles + cost_vel + cost_action), done

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> chex.Array:
        return self.action_array[action.squeeze()] * self.max_torque

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.concatenate([state.theta, state.theta_dot])

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        n = self.n_links
        return EnvState(theta=obs[:n], theta_dot=obs[n:], time=-1)

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        # compute link endpoints for a trajectory: theta trajectory shape (T, n_links)
        theta_traj = jnp.asarray(trajectory_state.theta)      # can be (T, n) or (n,) single time
        time_axis = trajectory_state.time

        # Ensure theta_traj is (T, n)
        if theta_traj.ndim == 1:
            theta_traj = theta_traj[jnp.newaxis, ...]
        T = theta_traj.shape[0]

        def endpoints_for_frame(theta_frame):
            # theta_frame: (n,)
            abs_angles = jnp.cumsum(theta_frame)  # absolute orientation of each link
            xs = []
            ys = []
            x, y = 0.0, 0.0
            for ang in abs_angles:
                dx = -self.length * jnp.sin(ang)
                dy = self.length * jnp.cos(ang)
                x = x + dx
                y = y + dy
                xs.append(x)
                ys.append(y)
            return jnp.array(xs), jnp.array(ys)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(self.name)
        reach = self.length * self.n_links
        ax.set_xlim(-reach * 1.2, reach * 1.2)
        ax.set_ylim(-reach * 1.2, reach * 1.2)
        ax.set_aspect('equal')
        ax.grid(True)

        # initial coordinates
        xs0, ys0 = endpoints_for_frame(theta_traj[0])
        line, = ax.plot([0.0] + xs0.tolist(), [0.0] + ys0.tolist(), lw=3, c='k', marker='o')
        circle_patches = []
        bob_radius = 0.08
        for (x_i, y_i) in zip(xs0, ys0):
            circle = plt.Circle((float(x_i), float(y_i)), bob_radius, fc='r', zorder=3)
            ax.add_patch(circle)
            circle_patches.append(circle)

        def update(frame):
            xs, ys = endpoints_for_frame(theta_traj[frame])
            line.set_data([0.0] + xs.tolist(), [0.0] + ys.tolist())
            for (circle, x_i, y_i) in zip(circle_patches, xs, ys):
                circle.set_center((float(x_i), float(y_i)))
            return (line, *circle_patches)

        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=T,
                                       interval=self.dt * 1000,
                                       blit=True
                                       )
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()
    # TODO sort out render_traj as unsure it actually works as of now

    @property
    def name(self) -> str:
        return "NPendulum-v0"

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

    def observation_space(self) -> spaces.Box:
        high = jnp.concatenate([jnp.ones((self.n_links,)) * jnp.pi, jnp.ones((self.n_links,)) * self.max_speed])
        return spaces.Box(-high, high, (2 * self.n_links,), dtype=jnp.float32)

    def reward_space(self) -> spaces.Box:
        # TODO actually add in the reward space bounds as unsure at the moment
        return spaces.Box(-100.0, 0.0, (()), dtype=jnp.float32)


class NPendulumCSCA(NPendulumCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_torque, self.max_torque).squeeze()

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_torque, self.max_torque, shape=(1,))



