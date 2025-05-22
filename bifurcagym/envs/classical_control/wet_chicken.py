"""
Based off the following: https://github.com/LAVA-LAB/improved_spi/blob/main/wetChicken.py
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
    # Implements the 2-dimensional Wet Chicken benchmark from 'Efficient Uncertainty Propagation for
    # Reinforcement Learning with Limited Data' by Alexander Hans and Steffen Udluft

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.length: float = 5.0
        self.width: float = 5.0
        self.max_turbulence: float = 3.5
        self.max_velocity: float = 3.0

        self.max_action: float = 1.0
        self.horizon: int = 200

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action)

        y_hat = state.y + (action[1] - 1.0) + self._velocity(state) + self._turbulence(state) * jrandom.uniform(key, minval=-1, maxval=1)
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

    def _velocity(self, state):
        return self.max_velocity * state.x / self.width

    def _turbulence(self, state):
        return self.max_turbulence - self._velocity(state)

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
        return -(self.length - state_tp1.y)  # TODO pretty sure its the new state y_t

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_action, self.max_action)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array([state.x, state.y])

    def get_state(self, obs: chex.Array) -> EnvState:
        return EnvState(x=obs[0], y=obs[1], time=-1)

    def is_done(self, state: EnvState) -> chex.Array:
        return jnp.array(False)  # TODO not sure about this as pretty sure it auto resets

    @property
    def name(self) -> str:
        return "WetChicken-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_action, self.max_action, shape=(2,))

    def observation_space(self) -> spaces.Box:
        low = jnp.array([0, 0])
        high = jnp.array([self.length, self.width])  # TODO check the ordering
        return spaces.Box(low, high, (2,))


class WetChickenCSDA(WetChickenCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: chex.Array = jnp.array(((0.0, 0.0),
                                                   (-1.0, 0.0),
                                                   (-2.0, 0.0),
                                                   (0.0, -1.0),
                                                   (0.0, 1.0),
                                                   ))

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action]

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))


    # TODO add in WetChicken Discrete State


# class WetChickenDSDA(WetChickenCSDA):
#     def __init__(self, **env_kwargs):
#         super().__init__(**env_kwargs)
#
#     # if self.discrete:
#     #     action_coordinates = list(ACTION_TRANSLATOR.values())[action]
#     #     y_hat = self._state[1] + action_coordinates[1] + self._velocity() + self._turbulence() * np.random.uniform(
#     #         -1, 1)
#     #     x_hat = self._state[0] + action_coordinates[0]
#     #     x_hat = round(x_hat)
#     #     y_hat = round(y_hat)
#     # else:
#
#     def observation_space(self) -> spaces.Discrete:
#         return spaces.Discrete(2)  # TODO what is this actually


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
    plot_2d_stuff()