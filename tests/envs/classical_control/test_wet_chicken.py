def test_wetchicken():
    env = WetChicken()
    n_tests = 100
    for _ in range(n_tests):
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs)
    done = False
    env.reset()
    for _ in range(env.horizon):
        action = env.action_space.sample()
        n, r, done, info = env.step(action)
        if done:
            break
    print("passed")


def plot_some_stuff():
    # Create environment
    env = WetChicken(seed=42)

    # Collect data over multiple episodes
    n_episodes = 1000
    steps_per_episode = 200  # Fixed number of steps per episode
    velocity_data = []
    turbulence_data = []
    reward_data = []
    positions = []

    for _ in range(n_episodes):
        obs = env.reset()

        for step in range(steps_per_episode):
            # Store current state data
            positions.append(obs[1])  # y position
            velocity = env.max_velocity * obs[1] / env.width
            velocity_data.append(velocity)

            # Get turbulence for current state
            turbulence = env._turbulence()  # Note: accessing protected method for visualization
            turbulence_data.append(turbulence)

            # Get reward
            reward = env._get_reward()
            reward_data.append(reward)

            # Take random action
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)

    # Convert to numpy arrays
    positions = np.array(positions)
    velocity_data = np.array(velocity_data)
    turbulence_data = np.array(turbulence_data)
    reward_data = np.array(reward_data)

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Velocity vs Position
    ax1.scatter(positions, velocity_data, alpha=0.1, color='blue', label='Observed')
    ax1.set_xlabel('Position (y)')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Velocity vs Position')
    ax1.grid(True)

    # Theoretical velocity line
    y_pos = np.linspace(0, env.width, 100)
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
    print(f"Average velocity: {np.mean(velocity_data):.2f}")
    print(f"Average turbulence: {np.mean(turbulence_data):.2f}")
    print(f"Average reward: {np.mean(reward_data):.2f}")
    print(f"Max position reached: {np.max(positions):.2f}")


def plot_2d_stuff():
    # Create environment
    env = WetChicken(seed=42, discrete=False)

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
        obs = env.reset()
        for step in range(steps_per_episode):
            # Get current x, y position
            x, y = obs

            # Find grid cell
            x_idx = int(np.clip(x * (grid_size - 1) / env.width, 0, grid_size - 1))
            y_idx = int(np.clip(y * (grid_size - 1) / env.length, 0, grid_size - 1))

            # Calculate values
            velocity = env._velocity()  # env.y_hat
            turbulence = env._turbulence()

            # Take random action
            action = env.action_space.sample()
            obs, reward, _, _ = env.step(action)

            # Update grids
            velocity_grid[x_idx, y_idx] += velocity
            turbulence_grid[x_idx, y_idx] += turbulence
            reward_grid[x_idx, y_idx] += reward
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
    visit_count_log = np.log1p(visit_count)  # log1p to handle zeros
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
    print(f"Percentage of grid cells visited: {100 * np.sum(visit_count > 0) / (grid_size * grid_size):.1f}%")
    print(f"Average visits per cell (where visited): {np.mean(visit_count[visit_count > 0]):.1f}")
    print(f"Maximum visits to a single cell: {np.max(visit_count):.0f}")


if __name__ == "__main__":
    test_wetchicken()
    # plot_some_stuff()
    # plot_2d_stuff()