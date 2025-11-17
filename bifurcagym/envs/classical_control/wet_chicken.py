"""
Based off the following: https://github.com/LAVA-LAB/improved_spi/blob/main/wetChicken.py

A 2D extension from: "https://www.researchgate.net/publication/221079849_Efficient_Uncertainty_Propagation_for_Reinforcement_Learning_with_Limited_Data"
"""

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
    x: chex.Array
    y: chex.Array
    time: int


class WetChickenCSCA(base_env.BaseEnvironment):

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.length: float = 5.0
        self.width: float = 5.0
        self.max_turbulence: float = 3.5
        self.turbulence_noise:float = 1.0
        self.max_velocity: float = 3.0

        self.max_action: float = 1.0
        self.horizon: int = 200

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        y_hat = state.y + action[1] + self._velocity(state) + self._turbulence(state, key)
        x_hat = state.x + action[0]

        # if y_hat >= self.length or x_hat < 0:
        #     x_new = 0
        # elif x_hat >= self.width:
        #     x_new = self.width
        # else:
        #     x_new = x_hat

        x_new_cond1 = jnp.where(x_hat >= self.width, self.width, x_hat)
        x_new = jnp.where(jnp.logical_or(y_hat >= self.length, x_hat < 0), 0.0, x_new_cond1)

        # if y_hat >= self.length:
        #     y_new = 0
        # elif y_hat < 0:
        #     y_new = 0
        # else:
        #     y_new = y_hat

        y_new = jnp.where(jnp.logical_or(y_hat >= self.length, y_hat < 0.0), 0.0, y_hat)

        new_state = EnvState(x=x_new, y=y_new, time=state.time+1)

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,  # TODO check reward is the correct way around
                self.is_done(new_state),
                {})

    def _velocity(self, state: EnvState) -> chex.Array:
        return self.max_velocity * state.x / self.width

    def _turbulence(self, state: EnvState, key: chex.PRNGKey) -> chex.Array:
        return (self.max_turbulence - self._velocity(state)) * jrandom.uniform(key,
                                                                               minval=-self.turbulence_noise,
                                                                               maxval=self.turbulence_noise)

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        state = EnvState(x=jnp.zeros(()),
                         y=jnp.zeros(()),
                         time=0)

        return self.get_obs(state), state

    def reward_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> chex.Array:
        return -(self.length - state_tp1.y)

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_action, self.max_action)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array([state.x, state.y])

    def get_state(self, obs: chex.Array) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], time=-1)

    def is_done(self, state: EnvState) -> chex.Array:
        return jnp.array(False)  # a continuous task as the environment auto resets as part of the setup

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.colors import LogNorm, Normalize

        grid_resolution = 50
        x_grid = jnp.linspace(0, self.width, grid_resolution)
        y_grid = jnp.linspace(0, self.length, grid_resolution)
        X, Y = jnp.meshgrid(x_grid, y_grid)

        vx_grid = jnp.linspace(0, self.width, 8)
        vy_grid = jnp.linspace(0, self.length, 6)
        VX, VY = jnp.meshgrid(vx_grid, vy_grid)

        def turb_plots(x, y, key):
            state = EnvState(x=x, y=y, time=-1)
            return self._turbulence(state, key)

        def vel_plots(x, y):
            state = EnvState(x=x, y=y, time=-1)
            return self._velocity(state)

        key = jrandom.key(0)
        batch_key = jrandom.split(key, grid_resolution * grid_resolution).reshape(grid_resolution, grid_resolution)
        turb_plot = jax.vmap(jax.vmap(turb_plots, in_axes=(0, None, 0)), in_axes=(None, 0, 0))(x_grid, y_grid, batch_key)
        vel_plot = jax.vmap(jax.vmap(vel_plots, in_axes=(0, None)), in_axes=(None, 0))(vx_grid, vy_grid)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(self.name)
        ax.set_xlim(0, self.width)
        ax.set_xlabel("X")
        ax.set_ylim(0, self.length)
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        ax.axhline(y=self.length, color="blue", label="Waterfall", linewidth=5)
        ax.legend(bbox_to_anchor=(1.4, 1.03))
        ax.grid(True)

        # norm_turb = LogNorm(vmin=turb_plot.min(), vmax=turb_plot.max())
        background_plot = ax.pcolormesh(X, Y, turb_plot, cmap='RdBu', shading='auto', zorder=-2, alpha=0.4)
        cbar = fig.colorbar(background_plot, ax=ax, orientation='vertical', shrink=0.75,
                            label='Turbulence Value' if 'vel_plot' in locals() else 'Turbulence Value')

        U = jnp.zeros_like(vel_plot)
        V = vel_plot
        magnitude = jnp.sqrt(U ** 2 + V ** 2)

        Q = ax.quiver(VX, VY, U, V,
                      magnitude,
                      cmap='summer_r',
                      angles='xy', scale_units='xy', scale=3, zorder=-1)

        cbar = fig.colorbar(Q, ax=ax, orientation='vertical', pad=0.05, shrink=0.75)
        cbar.set_label('Velocity Magnitude')
        # qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='axes')

        line, = ax.plot([], [], 'r-', lw=1.5, label='Agent Trail')
        dot, = ax.plot([], [], color="purple", marker="o", markersize=12, label='Current State')

        agent_path_history = jnp.array(((0.0,), (0.0,)))

        def update(frame):
            global agent_path_history

            x = jnp.expand_dims(trajectory_state.x[frame], axis=0)
            y = jnp.expand_dims(trajectory_state.y[frame], axis=0)

            if x == 0.0 and y == 0.0:
                agent_path_history = jnp.array(((0.0,), (0.0,)))
            else:
                xy = jnp.concatenate((jnp.expand_dims(x, 0), jnp.expand_dims(y, 0)))
                agent_path_history = jnp.concatenate((agent_path_history, xy), axis=-1)

            dot.set_data(x, y)

            line.set_data(agent_path_history[0], agent_path_history[1])

            return  line, dot

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=len(trajectory_state.time),
                                       interval=600,
                                       blit=True
                                       )
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "WetChicken-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_action, self.max_action, shape=(2,))

    def observation_space(self) -> spaces.Box:
        low = jnp.array([0, 0])
        high = jnp.array([self.width, self.length])
        return spaces.Box(low, high, (2,))

    def reward_space(self) -> spaces.Box:
        return spaces.Box(-self.length, 0, (()), dtype=jnp.float32)


class WetChickenCSDA(WetChickenCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: chex.Array = jnp.array(((0.0, 0.0),   # Drift
                                                   (0.0, -1.0),  # Hold
                                                   (0.0, -2.0),  # Paddle back
                                                   (1.0, 0.0),   # Right
                                                   (-1.0, 0.0),  # Left
                                                   ))

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action.squeeze()]

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))


class WetChickenDSDA(WetChickenCSDA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.length: int = 5
        self.width: int = 5

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        y_hat = jnp.round(state.y + action[1] + self._velocity(state) + self._turbulence(state, key))
        x_hat = jnp.round(state.x + action[0])

        x_new_cond1 = jnp.where(x_hat > (self.width - 1), (self.width - 1), x_hat)
        x_new = jnp.where(jnp.logical_or(y_hat >= self.length, x_hat < 0), 0, x_new_cond1)

        y_new = jnp.where(jnp.logical_or(y_hat >= self.length, y_hat < 0), 0, y_hat)

        new_state = EnvState(x=x_new, y=y_new, time=state.time+1)

        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {})

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array([state.x + state.y * self.width,])

    def get_state(self, obs: chex.Array) -> EnvState:
        y = obs // self.width
        x = obs % self.width
        return EnvState(x=x.squeeze(), y=y.squeeze(), time=-1)

    def observation_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.width * self.length)


def plot_some_stuff():
    import matplotlib.pyplot as plt
    key = jrandom.PRNGKey(0)
    env = WetChickenCSCA()

    # Collect data over multiple episodes
    n_episodes = 10#00
    steps_per_episode = 200  # Fixed number of steps per episode
    velocity_data = []
    turbulence_data = []
    reward_data = []
    positions = []

    for _ in range(n_episodes):
        key, _key = jrandom.split(key)
        obs, env_state = env.reset(_key)

        for step in range(steps_per_episode):
            # Store current state data
            positions.append(obs[1])  # y position
            velocity = env.max_velocity * obs[1] / env.width
            velocity_data.append(velocity)

            # Get turbulence for current state
            turbulence = env._turbulence(env_state)  # Note: accessing protected method for visualization
            turbulence_data.append(turbulence)

            # Take random action
            key, _key = jrandom.split(key)
            action = env.action_space().sample(_key)
            key, _key = jrandom.split(key)
            obs, delta_obs, env_state, rew, done, info = env.step(action, env_state, _key)
            reward_data.append(rew)

    # Convert to numpy arrays
    positions = jnp.array(positions)
    velocity_data = jnp.array(velocity_data)
    turbulence_data = jnp.array(turbulence_data)
    reward_data = jnp.array(reward_data)

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Velocity vs Position
    ax1.scatter(positions, velocity_data, alpha=0.1, color='blue', label='Observed')
    ax1.set_xlabel('Position (y)')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Velocity vs Position')
    ax1.grid(True)

    # Theoretical velocity line
    y_pos = jnp.linspace(0, env.width, 100)
    theoretical_velocity = env.max_velocity * y_pos / env.width
    ax1.plot(y_pos, theoretical_velocity, 'r--', label='Theoretical')
    ax1.legend()

    # Plot 2: Turbulence vs Position
    ax2.scatter(positions, turbulence_data, alpha=0.1, color='green')
    ax2.set_xlabel('Position (y)')
    ax2.set_ylabel('Turbulence')
    ax2.set_title('Turbulence vs Position')
    ax2.grid(True)

    # Add theoretical turbulence bounds
    theoretical_velocity = env.max_velocity * y_pos / env.width
    max_turbulence = env.max_turbulence - theoretical_velocity
    ax2.plot(y_pos, max_turbulence, 'r--', label='Max Turbulence')
    ax2.plot(y_pos, -max_turbulence, 'r--', label='Min Turbulence')
    ax2.legend()

    # Plot 3: Reward vs Position
    ax3.scatter(positions, reward_data, alpha=0.1, color='purple')
    ax3.set_xlabel('Position (y)')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward vs Position')
    ax3.grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"\nStatistics over {n_episodes} episodes:")
    print(f"Average velocity: {jnp.mean(velocity_data):.2f}")
    print(f"Average turbulence: {jnp.mean(turbulence_data):.2f}")
    print(f"Average reward: {jnp.mean(reward_data):.2f}")
    print(f"Max position reached: {jnp.max(positions):.2f}")


def plot_2d_stuff():
    import matplotlib.pyplot as plt
    import numpy as np
    key = jrandom.PRNGKey(0)
    env = WetChickenCSCA()

    # Parameters for data collection
    n_episodes = 400
    steps_per_episode = 2000

    # Create grid for storing average values
    grid_size = 50  # Resolution of our grid
    x_grid = np.linspace(0, env.width, grid_size)
    y_grid = np.linspace(0, env.length, grid_size)
    velocity_grid = np.zeros((grid_size, grid_size))
    turbulence_grid = np.zeros((grid_size, grid_size))
    reward_grid = np.zeros((grid_size, grid_size))
    visit_count = np.zeros((grid_size, grid_size))

    for episode in range(n_episodes):
        key, _key = jrandom.split(key)
        obs, env_state = env.reset(_key)

        for step in range(steps_per_episode):
            # Get current x, y position
            x, y = obs

            # Find grid cell
            x_idx = int(jnp.clip(x * (grid_size - 1) / env.width, 0, grid_size - 1))
            y_idx = int(jnp.clip(y * (grid_size - 1) / env.length, 0, grid_size - 1))

            # Calculate values
            velocity = env._velocity(env_state)  # env.y_hat
            turbulence = env._turbulence(env_state)

            # Take random action
            key, _key = jrandom.split(key)
            action = env.action_space().sample(_key)
            key, _key = jrandom.split(key)
            with jax.disable_jit(disable=False):
                obs, delta_obs, env_state, rew, done, info = env.step(action, env_state, _key)

            # Update grids
            velocity_grid[x_idx, y_idx] += velocity
            turbulence_grid[x_idx, y_idx] += turbulence
            reward_grid[x_idx, y_idx] += rew
            visit_count[x_idx, y_idx] += 1

    # Average the grids where visited
    mask = visit_count > 0
    velocity_grid[mask] /= visit_count[mask]
    turbulence_grid[mask] /= visit_count[mask]
    reward_grid[mask] /= visit_count[mask]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

    # Plot 1: Velocity Heatmap
    im1 = ax1.imshow(velocity_grid.T, origin='lower', extent=[0, env.width, 0, env.length],
                     aspect='auto', cmap='viridis')
    ax1.set_title('Velocity across X-Y plane')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    plt.colorbar(im1, ax=ax1, label='Velocity')

    # Plot 2: Turbulence Heatmap
    im2 = ax2.imshow(turbulence_grid.T, origin='lower', extent=[0, env.width, 0, env.length],
                     aspect='auto', cmap='RdBu')
    ax2.set_title('Turbulence across X-Y plane')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax2, label='Turbulence')

    # Plot 3: Reward Heatmap
    im3 = ax3.imshow(reward_grid.T, origin='lower', extent=[0, env.width, 0, env.length],
                     aspect='auto', cmap='plasma')
    ax3.set_title('Reward across X-Y plane')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    plt.colorbar(im3, ax=ax3, label='Reward')

    # Plot 4: Visit Count Heatmap (log scale for better visualization)
    visit_count_log = jnp.log1p(visit_count)  # log1p to handle zeros
    im4 = ax4.imshow(visit_count_log.T, origin='lower', extent=[0, env.width, 0, env.length],
                     aspect='auto', cmap='YlOrRd')
    ax4.set_title('Visit Count across X-Y plane (log scale)')
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    plt.colorbar(im4, ax=ax4, label='Log(Visit Count + 1)')

    plt.tight_layout()
    plt.show()

    # Print some statistics about the coverage
    print(f"\nStatistics over {n_episodes} episodes ({n_episodes * steps_per_episode} total steps):")
    print(f"Percentage of grid cells visited: {100 * jnp.sum(visit_count > 0) / (grid_size * grid_size):.1f}%")
    print(f"Average visits per cell (where visited): {jnp.mean(visit_count[visit_count > 0]):.1f}")
    print(f"Maximum visits to a single cell: {jnp.max(visit_count):.0f}")


if __name__ == "__main__":
    # plot_some_stuff()
    # plot_2d_stuff()

    key = jrandom.PRNGKey(0)
    env = WetChickenCSCA()

    num_steps = 200  # Fixed number of steps per episode

    key, _key = jrandom.split(key)
    obs, env_state = env.reset(_key)

    def _loop_func(runner_state, unused):
        obs, env_state, key = runner_state
        key, _key = jrandom.split(key)
        action = env.action_space().sample(_key)
        key, _key = jrandom.split(key)
        nobs, delta_obs, next_env_state, rew, done, info = env.step(action, env_state, _key)

        return (nobs, next_env_state, key), env_state

    _, traj = jax.lax.scan(_loop_func, (obs, env_state, key), None, num_steps)

    env.render_traj(traj)