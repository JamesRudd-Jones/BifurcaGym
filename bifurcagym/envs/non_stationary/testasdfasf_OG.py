import numpy as np
import time
import os


class ChaoticSailingEnv:
    """
    A simple grid-world RL environment simulating a boat navigating dynamic currents.
    This version uses a simplified, grid-based fluid dynamics solver for realistic currents.

    The environment's dynamics can be configured to be:
    - Stationary: Currents are constant (if non-stationarity is off).
    - Stochastic: Random gusts affect movement.
    - Non-Stationary: Currents change over time.
    - Chaotic (Fluid Dynamics): Currents are a complex velocity field simulated
      using a stable fluid dynamics solver, creating realistic eddies and flows.
    """

    def __init__(self, config=None):
        default_config = {
            'grid_size': 15,
            'is_stochastic': True,
            'is_non_stationary': True,
            'is_chaotic': True,  # Use the fluid dynamics solver
            'stochasticity_level': 0.1,
            'current_strength': 1.0,
            'fluid_viscosity': 0.0001,  # How thick the fluid is
            'fluid_dt': 0.1,  # Simulation time step
        }
        if config:
            default_config.update(config)

        # --- Core Parameters ---
        self.grid_size = default_config['grid_size']
        self.is_stochastic = default_config['is_stochastic']
        self.is_non_stationary = default_config['is_non_stationary']
        self.is_chaotic = default_config['is_chaotic']
        self.stochasticity_level = default_config['stochasticity_level']
        self.current_strength = default_config['current_strength']

        # --- Fluid Dynamics Parameters ---
        self.visc = default_config['fluid_viscosity']
        self.dt = default_config['fluid_dt']
        self.N = self.grid_size

        # Velocity field (vx, vy) and previous velocity field
        self.vx = np.zeros((self.N, self.N))
        self.vy = np.zeros((self.N, self.N))
        self.vx0 = np.zeros((self.N, self.N))
        self.vy0 = np.zeros((self.N, self.N))

        # --- Action and Observation Space ---
        self.action_space_n = 5
        self._action_to_direction = {
            0: np.array([-1, 0]), 1: np.array([0, 1]), 2: np.array([1, 0]),
            3: np.array([0, -1]), 4: np.array([0, 0]),
        }
        self.observation_space_shape = (2,)

        # --- Environment State ---
        self.start_pos = np.array([self.grid_size - 1, 0])
        self.goal_pos = np.array([0, self.grid_size - 1])
        self.agent_pos = None
        self.time_step = 0

        self.reset()

    # ==============================================================================
    # FLUID DYNAMICS SOLVER (Based on Jos Stam's "Real-Time Fluid Dynamics for Games")
    # ==============================================================================

    def _set_bnd(self, b, x):
        """Set boundary conditions for the fluid grid."""
        x[0, :] = -x[1, :] if b == 1 else x[1, :]
        x[self.N - 1, :] = -x[self.N - 2, :] if b == 1 else x[self.N - 2, :]
        x[:, 0] = -x[:, 1] if b == 2 else x[:, 1]
        x[:, self.N - 1] = -x[:, self.N - 2] if b == 2 else x[:, self.N - 2]
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, self.N - 1] = 0.5 * (x[1, self.N - 1] + x[0, self.N - 2])
        x[self.N - 1, 0] = 0.5 * (x[self.N - 2, 0] + x[self.N - 1, 1])
        x[self.N - 1, self.N - 1] = 0.5 * (x[self.N - 2, self.N - 1] + x[self.N - 1, self.N - 2])

    def _lin_solve(self, b, x, x0, a, c, iter=4):
        """Iterative linear solver using Gauss-Seidel relaxation."""
        c_recip = 1.0 / c
        for _ in range(iter):
            # The core of the solver: update each cell based on its neighbors
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])) * c_recip
            self._set_bnd(b, x)

    def _diffuse(self, b, x, x0, iter=4):
        """Diffusion step: spread velocities based on viscosity."""
        a = self.dt * self.visc * (self.N - 2) * (self.N - 2)
        self._lin_solve(b, x, x0, a, 1 + 4 * a, iter)

    def _advect(self, b, d, d0, velocX, velocY):
        """Advection step: move the fluid along its own velocity field."""
        dtx = self.dt * (self.N - 2)
        dty = self.dt * (self.N - 2)

        i, j = np.meshgrid(np.arange(1, self.N - 1), np.arange(1, self.N - 1), indexing='ij')
        tmp1 = dtx * velocX[i, j]
        tmp2 = dty * velocY[i, j]
        x = i.astype(float) - tmp1
        y = j.astype(float) - tmp2

        x = np.clip(x, 0.5, self.N - 1.5)
        y = np.clip(y, 0.5, self.N - 1.5)

        i0, i1 = x.astype(int), x.astype(int) + 1
        j0, j1 = y.astype(int), y.astype(int) + 1

        s1 = x - i0
        s0 = 1 - s1
        t1 = y - j0
        t0 = 1 - t1

        d[i, j] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                   s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))
        self._set_bnd(b, d)

    def _project(self, velocX, velocY, p, div, iter=4):
        """Projection step: enforce mass conservation (incompressibility)."""
        div[1:-1, 1:-1] = -0.5 * (velocX[2:, 1:-1] - velocX[:-2, 1:-1] +
                                  velocY[1:-1, 2:] - velocY[1:-1, :-2]) / self.N
        p.fill(0)
        self._set_bnd(0, div)
        self._set_bnd(0, p)
        self._lin_solve(0, p, div, 1, 4, iter)

        velocX[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * self.N
        velocY[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * self.N
        self._set_bnd(1, velocX)
        self._set_bnd(2, velocY)

    def _add_random_forces(self):
        """Injects random forces into the fluid to keep it dynamic."""
        if self.time_step % 5 == 0:  # Add force every 5 steps
            # Add a strong, directional force at a random location
            force_x = (np.random.rand() - 0.5) * 50 * self.current_strength
            force_y = (np.random.rand() - 0.5) * 50 * self.current_strength
            px, py = np.random.randint(self.N // 4, 3 * self.N // 4, 2)
            self.vx[px, py] += force_x
            self.vy[px, py] += force_y

    def _update_fluid_dynamics(self):
        """The main fluid simulation step."""
        self._add_random_forces()

        self.vx0, self.vy0 = self.vx.copy(), self.vy.copy()

        self._diffuse(1, self.vx0, self.vx)
        self._diffuse(2, self.vy0, self.vy)

        self._project(self.vx0, self.vy0, self.vx, self.vy)

        self.vx, self.vy = self.vx0.copy(), self.vy0.copy()

        self._advect(1, self.vx, self.vx0, self.vx0, self.vy0)
        self._advect(2, self.vy, self.vy0, self.vx0, self.vy0)

        self._project(self.vx, self.vy, self.vx0, self.vy0)

    # ==============================================================================
    # RL ENVIRONMENT INTERFACE
    # ==============================================================================

    def _get_current_at_pos(self, pos):
        """Interpolates the fluid velocity at the agent's continuous position."""
        y, x = pos
        x = np.clip(x, 0, self.N - 1.001)
        y = np.clip(y, 0, self.N - 1.001)

        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1

        s1 = x - x0
        s0 = 1 - s1
        t1 = y - y0
        t0 = 1 - t1

        vx = (s0 * (t0 * self.vx[y0, x0] + t1 * self.vx[y1, x0]) +
              s1 * (t0 * self.vx[y0, x1] + t1 * self.vx[y1, x1]))
        vy = (s0 * (t0 * self.vy[y0, x0] + t1 * self.vy[y1, x0]) +
              s1 * (t0 * self.vy[y0, x1] + t1 * self.vy[y1, x1]))

        return np.array([vy, vx])  # Return as [row_change, col_change]

    def step(self, action):
        if self.agent_pos is None:
            raise RuntimeError("You must call reset() before calling step()")

        # 1. Get intended movement
        intended_move = self._action_to_direction[action]
        total_move = intended_move.astype(float)

        # 2. Apply stochastic gusts
        if self.is_stochastic:
            gust = np.random.randn(2) * self.stochasticity_level
            total_move += gust

        # 3. Apply the current from the fluid field
        current_at_pos = self._get_current_at_pos(self.agent_pos)
        total_move += current_at_pos * self.current_strength

        # 4. Update agent position and clip to boundaries
        self.agent_pos += total_move
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)

        # 5. Calculate reward
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -distance_to_goal
        done = False
        if distance_to_goal < 1.0:
            reward += 100
            done = True

        # 6. Update the environment dynamics for the next step
        if self.is_non_stationary:
            if self.is_chaotic:
                self._update_fluid_dynamics()
            else:  # Fallback to simple sine wave if not chaotic
                freq = 0.1
                # This part is now less relevant but kept for comparison
                self.vx.fill(np.cos(self.time_step * freq * 0.5) * self.current_strength * 0.1)
                self.vy.fill(np.sin(self.time_step * freq) * self.current_strength * 0.1)

        self.time_step += 1
        info = {'current_at_pos': current_at_pos}
        return np.round(self.agent_pos).astype(int), reward, done, info

    def reset(self):
        self.agent_pos = self.start_pos.copy().astype(float)
        self.time_step = 0
        self.vx.fill(0)
        self.vy.fill(0)
        return np.round(self.agent_pos).astype(int)

    def render(self, mode='human'):
        if mode == 'human':
            os.system('cls' if os.name == 'nt' else 'clear')
            grid = np.full((self.grid_size, self.grid_size), '.')

            agent_r, agent_c = np.round(self.agent_pos).astype(int)
            agent_r = np.clip(agent_r, 0, self.grid_size - 1)
            agent_c = np.clip(agent_c, 0, self.grid_size - 1)

            goal_r, goal_c = self.goal_pos

            grid[goal_r, goal_c] = 'G'
            grid[agent_r, agent_c] = 'B'

            current_at_pos = self._get_current_at_pos(self.agent_pos)
            print(f"Time: {self.time_step}")
            print(f"Current at Boat: [x={current_at_pos[1]:.2f}, y={current_at_pos[0]:.2f}]")
            print("\n".join(" ".join(row) for row in grid))
            print("-" * (self.grid_size * 2))


if __name__ == '__main__':
    config = {'grid_size': 20,
              'is_stochastic': False,
              'is_non_stationary': False,
              'is_chaotic': False,  # Use the new fluid solver
              'current_strength': 1.5,
              'stochasticity_level': 0.05,
              'fluid_viscosity': 0.00001
              }
    env = ChaoticSailingEnv(config=config)

    obs = env.reset()
    done = False
    total_reward = 0

    for step in range(500):
        if done:
            print(f"Goal reached in {step} steps! Total reward: {total_reward:.2f}")
            break

        action = np.random.randint(0, env.action_space_n)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        env.render()
        print(f"Action: {['Up', 'Right', 'Down', 'Left', 'Stay'][action]}, Reward: {reward:.2f}")
        time.sleep(0.05)

    if not done:
        print("Episode finished without reaching the goal.")

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import time
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# DATASTRUCTURES FOR STATE AND PARAMS
# ==============================================================================

@dataclass
class EnvParams:
    """Static environment parameters that do not change during a rollout."""
    grid_size: int = 15
    is_stochastic: bool = True
    is_non_stationary: bool = True
    is_chaotic: bool = True
    stochasticity_level: float = 0.1
    current_strength: float = 1.0
    fluid_viscosity: float = 0.0001
    fluid_dt: float = 0.1
    action_map: jnp.ndarray = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])
    start_pos: jnp.ndarray = jnp.array([14, 0])
    goal_pos: jnp.ndarray = jnp.array([0, 14])
    max_steps_in_episode: int = 200


@dataclass
class EnvState:
    """Dynamic environment state that changes at each step."""
    agent_pos: jnp.ndarray
    vx: jnp.ndarray  # Fluid velocity field (x-component)
    vy: jnp.ndarray  # Fluid velocity field (y-component)
    time_step: int
    key: jax.random.PRNGKey


# ==============================================================================
# JAX-BASED FLUID DYNAMICS SOLVER
# ==============================================================================

@partial(jax.jit, static_argnums=(0,))
def _set_bnd(b: int, x: jnp.ndarray) -> jnp.ndarray:
    """Set boundary conditions for the fluid grid in a JAX-compatible way."""
    N = x.shape[0]
    x = x.at[0, :].set(jnp.where(b == 1, -x[1, :], x[1, :]))
    x = x.at[N - 1, :].set(jnp.where(b == 1, -x[N - 2, :], x[N - 2, :]))
    x = x.at[:, 0].set(jnp.where(b == 2, -x[:, 1], x[:, 1]))
    x = x.at[:, N - 1].set(jnp.where(b == 2, -x[:, N - 2], x[:, N - 2]))

    x = x.at[0, 0].set(0.5 * (x[1, 0] + x[0, 1]))
    x = x.at[0, N - 1].set(0.5 * (x[1, N - 1] + x[0, N - 2]))
    x = x.at[N - 1, 0].set(0.5 * (x[N - 2, 0] + x[N - 1, 1]))
    x = x.at[N - 1, N - 1].set(0.5 * (x[N - 2, N - 1] + x[N - 1, N - 2]))
    return x


@partial(jax.jit, static_argnums=(0, 4, 5))
def _lin_solve(b: int, x: jnp.ndarray, x0: jnp.ndarray, a: float, c: float, iter_count: int) -> jnp.ndarray:
    """Iterative linear solver using Gauss-Seidel relaxation."""
    c_recip = 1.0 / c

    def loop_body(i, val):
        x_new = val
        x_new = x_new.at[1:-1, 1:-1].set(
            (x0[1:-1, 1:-1] + a * (x_new[:-2, 1:-1] + x_new[2:, 1:-1] + x_new[1:-1, :-2] + x_new[1:-1, 2:])) * c_recip
        )
        x_new = _set_bnd(b, x_new)
        return x_new

    x = lax.fori_loop(0, iter_count, loop_body, x)
    return x


@jax.jit
def _diffuse(params: EnvParams, b: int, x: jnp.ndarray, x0: jnp.ndarray) -> jnp.ndarray:
    """Diffusion step."""
    N = params.grid_size
    a = params.fluid_dt * params.fluid_viscosity * (N - 2) * (N - 2)
    return _lin_solve(b, x, x0, a, 1 + 4 * a, 4)


@jax.jit
def _advect(params: EnvParams, b: int, d: jnp.ndarray, d0: jnp.ndarray, velocX: jnp.ndarray,
            velocY: jnp.ndarray) -> jnp.ndarray:
    """Advection step."""
    N = params.grid_size
    dtx = params.fluid_dt * (N - 2)
    dty = params.fluid_dt * (N - 2)

    j, i = jnp.meshgrid(jnp.arange(1, N - 1), jnp.arange(1, N - 1), indexing='ij')

    x = i.astype(float) - dtx * velocX[i, j]
    y = j.astype(float) - dty * velocY[i, j]

    x = jnp.clip(x, 0.5, N - 1.5)
    y = jnp.clip(y, 0.5, N - 1.5)

    i0, i1 = x.astype(int), x.astype(int) + 1
    j0, j1 = y.astype(int), y.astype(int) + 1

    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1

    d_new = d.at[i, j].set(
        s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
        s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    )
    return _set_bnd(b, d_new)


@jax.jit
def _project(params: EnvParams, velocX: jnp.ndarray, velocY: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Projection step."""
    N = params.grid_size
    p = jnp.zeros((N, N))
    div = jnp.zeros((N, N))

    div = div.at[1:-1, 1:-1].set(
        -0.5 * (velocX[2:, 1:-1] - velocX[:-2, 1:-1] + velocY[1:-1, 2:] - velocY[1:-1, :-2]) / N)
    div = _set_bnd(0, div)

    p = _lin_solve(0, p, div, 1, 4, 4)

    velocX = velocX.at[1:-1, 1:-1].add(-0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * N)
    velocY = velocY.at[1:-1, 1:-1].add(-0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * N)

    velocX = _set_bnd(1, velocX)
    velocY = _set_bnd(2, velocY)
    return velocX, velocY


@jax.jit
def _add_random_forces(state: EnvState, params: EnvParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Injects random forces into the fluid."""
    key, subkey_force, subkey_pos = jax.random.split(state.key, 3)

    force = jax.random.uniform(subkey_force, shape=(2,), minval=-0.5, maxval=0.5) * 50 * params.current_strength
    pos = jax.random.randint(subkey_pos, shape=(2,), minval=params.grid_size // 4, maxval=3 * params.grid_size // 4)

    # Add force only on certain timesteps
    vx = lax.cond(state.time_step % 5 == 0,
                  lambda v: v.at[pos[0], pos[1]].add(force[0]),
                  lambda v: v,
                  state.vx)
    vy = lax.cond(state.time_step % 5 == 0,
                  lambda v: v.at[pos[1], pos[0]].add(force[1]),
                  lambda v: v,
                  state.vy)

    return vx, vy


@jax.jit
def _update_fluid_dynamics(state: EnvState, params: EnvParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The main fluid simulation step."""
    vx, vy = _add_random_forces(state, params)

    vx0, vy0 = vx, vy

    vx = _diffuse(params, 1, vx, vx0)
    vy = _diffuse(params, 2, vy, vy0)

    vx, vy = _project(params, vx, vy)

    vx0, vy0 = vx, vy

    vx = _advect(params, 1, vx, vx0, vx0, vy0)
    vy = _advect(params, 2, vy, vy0, vx0, vy0)

    vx, vy = _project(params, vx, vy)
    return vx, vy


# ==============================================================================
# GYMNAX-STYLE ENVIRONMENT
# ==============================================================================

class ChaoticSailingJax:
    def __init__(self, config: Optional[dict] = None):
        """Initializes the environment class, setting up static parameters."""
        # Use provided config or default EnvParams
        if config:
            self.params = EnvParams(**config)
        else:
            self.params = EnvParams()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @partial(jax.jit, static_argnames=("self",))
    def step(self, key: jax.random.PRNGKey, state: EnvState, action: int) -> Tuple[
        jnp.ndarray, EnvState, float, bool, dict]:
        """Environment step function."""
        # 1. Update dynamics if non-stationary
        vx, vy = lax.cond(
            self.params.is_non_stationary and self.params.is_chaotic,
            lambda s: _update_fluid_dynamics(s, self.params),
            lambda s: (s.vx, s.vy),
            state
        )

        # 2. Get agent's intended move and apply stochastic gusts
        key, gust_key = jax.random.split(key)
        intended_move = self.params.action_map[action]
        gust = jax.random.normal(gust_key, (2,)) * self.params.stochasticity_level
        total_move = intended_move.astype(float) + lax.cond(self.params.is_stochastic, lambda: gust,
                                                            lambda: jnp.zeros(2), None)

        # 3. Apply current from fluid field (interpolated)
        y, x = state.agent_pos
        x = jnp.clip(x, 0, self.params.grid_size - 1.001)
        y = jnp.clip(y, 0, self.params.grid_size - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        x1, y1 = x0 + 1, y0 + 1
        s1, t1 = x - x0, y - y0
        s0, t0 = 1 - s1, 1 - t1

        current_vx = (s0 * (t0 * vx[y0, x0] + t1 * vx[y1, x0]) + s1 * (t0 * vx[y0, x1] + t1 * vx[y1, x1]))
        current_vy = (s0 * (t0 * vy[y0, x0] + t1 * vy[y1, x0]) + s1 * (t0 * vy[y0, x1] + t1 * vy[y1, x1]))
        current_at_pos = jnp.array([current_vy, current_vx])

        total_move += current_at_pos * self.params.current_strength

        # 4. Update agent position and time
        new_agent_pos = jnp.clip(state.agent_pos + total_move, 0, self.params.grid_size - 1)
        time_step = state.time_step + 1

        # 5. Calculate reward and done flag
        distance_to_goal = jnp.linalg.norm(new_agent_pos - self.params.goal_pos)
        reward = -distance_to_goal

        done = (distance_to_goal < 1.0) | (time_step >= self.params.max_steps_in_episode)
        reward = jnp.where(distance_to_goal < 1.0, reward + 100, reward)

        # 6. Create new state and observation
        new_state = EnvState(agent_pos=new_agent_pos, vx=vx, vy=vy, time_step=time_step, key=key)
        obs = new_agent_pos.round().astype(jnp.int32)
        info = {"distance": distance_to_goal, "current_at_pos": current_at_pos}

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnames=("self",))
    def reset(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, EnvState]:
        """Reset the environment."""
        N = self.params.grid_size
        state = EnvState(
            agent_pos=self.params.start_pos.astype(float),
            vx=jnp.zeros((N, N)),
            vy=jnp.zeros((N, N)),
            time_step=0,
            key=key
        )
        obs = state.agent_pos.round().astype(jnp.int32)
        return obs, state


def render(state: EnvState, params: EnvParams):
    """Simple text-based rendering. Not JIT-compatible."""
    os.system('cls' if os.name == 'nt' else 'clear')
    grid = [["." for _ in range(params.grid_size)] for _ in range(params.grid_size)]

    agent_r, agent_c = state.agent_pos.round().astype(int)
    agent_r = jnp.clip(agent_r, 0, params.grid_size - 1)
    agent_c = jnp.clip(agent_c, 0, params.grid_size - 1)

    goal_r, goal_c = params.goal_pos

    grid[goal_r][goal_c] = 'G'
    grid[agent_r][agent_c] = 'B'

    print(f"Time: {state.time_step}")
    print("\n".join(" ".join(row) for row in grid))
    print("-" * (params.grid_size * 2))


if __name__ == '__main__':
    # --- DEMO: Run the JAX environment ---

    # Configuration for a chaotic, stochastic environment
    config = {
        'grid_size': 20,
        'start_pos': jnp.array([19, 0]),
        'goal_pos': jnp.array([0, 19]),
        'is_stochastic': True,
        'is_non_stationary': True,
        'is_chaotic': True,
        'current_strength': 1.8,
        'stochasticity_level': 0.05,
        'fluid_viscosity': 0.00001,
        'max_steps_in_episode': 300
    }

    env = ChaoticSailingJax(config)
    params = env.params

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)

    obs, state = env.reset(reset_key)

    # JIT-compile the step function for massive speedup
    jit_step = jax.jit(env.step)

    total_reward = 0

    for step_num in range(params.max_steps_in_episode):
        key, action_key, step_key = jax.random.split(key, 3)
        action = jax.random.randint(action_key, (), 0, 5)

        # Call the JIT-compiled step function
        obs, state, reward, done, info = jit_step(step_key, state, action)
        total_reward += reward

        # Rendering is done outside the JIT loop
        render(state, params)
        print(f"Step: {step_num}, Action: {action}, Reward: {reward:.2f}, Done: {done}")
        print(f"Distance to Goal: {info['distance']:.2f}")

        if done:
            print(f"\nEpisode finished after {step_num + 1} steps!")
            break

        time.sleep(0.05)

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.ndimage
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex
from functools import partial


@struct.dataclass
class EnvState(base_env.EnvState):
    """
    Represents the state of the environment.

    Attributes:
        x (chex.Array): The x-coordinate of the boat.
        y (chex.Array): The y-coordinate of the boat.
        time (int): The current timestep.
        key (chex.PRNGKey): JAX random key.
        fluid_u (chex.Array): The u-component (x-velocity) of the fluid grid.
        fluid_v (chex.Array): The v-component (y-velocity) of the fluid grid.
    """
    x: chex.Array
    y: chex.Array
    time: int
    key: chex.PRNGKey
    # New state for the fluid grid
    fluid_u: chex.Array
    fluid_v: chex.Array


class BoatInCurrentCSCA(base_env.BaseEnvironment):
    """
    A 2D environment simulating a boat in a current.
    The current can be modeled in several ways, including a chaotic fluid solver.
    """

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # Environment dimensions
        self.length: float = 15.0
        self.width: float = 15.0

        # Goal state
        self.goal_state = jnp.array((self.length, self.width))

        # Action and horizon limits
        self.max_action: float = 1.0
        self.horizon: int = 200

        # --- Fluid Dynamics Parameters for Chaotic Current ---
        self.is_chaotic: bool = True  # Set to True to use the fluid solver
        self.fluid_grid_size: int = 64  # Resolution of the fluid grid (NxN)
        self.fluid_dt: float = 0.2  # Timestep for the fluid simulation
        self.fluid_viscosity: float = 1e-6  # Viscosity of the fluid
        self.fluid_iterations: int = 20  # Number of iterations for the linear solver
        self.fluid_force: float = 5.0  # Strength of the external force driving the fluid

    # --------------------------------------------------------------------------------
    # JIT-compiled static methods for the fluid solver
    # Based on "Stable Fluids" by Jos Stam
    # --------------------------------------------------------------------------------

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _linear_solve(x: chex.Array, b: chex.Array, a: float, c: float, num_iterations: int) -> chex.Array:
        """
        Solves a linear system using Jacobi iteration with periodic boundary conditions.
        Used for diffusion and pressure projection.
        """

        def body_fun(_, val):
            x_prev = val
            # Neighbors are found using jnp.roll for periodic boundaries
            neighbors = (jnp.roll(x_prev, 1, axis=0) + jnp.roll(x_prev, -1, axis=0) +
                         jnp.roll(x_prev, 1, axis=1) + jnp.roll(x_prev, -1, axis=1))
            x_new = (b + a * neighbors) / c
            return x_new

        return jax.lax.fori_loop(0, num_iterations, body_fun, x)

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def _diffuse(field: chex.Array, viscosity: float, dt: float, grid_size: int, num_iterations: int) -> chex.Array:
        """Applies fluid diffusion (viscosity)."""
        a = dt * viscosity * grid_size * grid_size
        return BoatInCurrentCSCA._linear_solve(field, field, a, 1 + 4 * a, num_iterations)

    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4))
    def _advect(field: chex.Array, u: chex.Array, v: chex.Array, dt: float, grid_size: int) -> chex.Array:
        """Moves a quantity 'field' through the velocity field (u, v)."""
        N = grid_size
        x_coords, y_coords = jnp.meshgrid(jnp.arange(N), jnp.arange(N))

        # Trace back in time to find the source of the fluid
        back_x = x_coords - (dt * u * N)
        back_y = y_coords - (dt * v * N)

        coords = jnp.stack([back_y, back_x], axis=0)

        # Sample from the original field at the back-traced coordinates
        return jax.scipy.ndimage.map_coordinates(field, coords, order=1, mode='wrap')

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _project(u: chex.Array, v: chex.Array, num_iterations: int) -> Tuple[chex.Array, chex.Array]:
        """Enforces fluid incompressibility."""
        # Calculate divergence using central differences
        div = -0.5 * (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1) +
                      jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0))

        # Solve Poisson equation for pressure
        p = jnp.zeros_like(div)
        p = BoatInCurrentCSCA._linear_solve(p, -div, 1.0, 4.0, num_iterations)

        # Subtract the pressure gradient from the velocity field
        u_new = u - 0.5 * (jnp.roll(p, -1, axis=1) - jnp.roll(p, 1, axis=1))
        v_new = v - 0.5 * (jnp.roll(p, -1, axis=0) - jnp.roll(p, 1, axis=0))

        return u_new, v_new

    # --------------------------------------------------------------------------------
    # Core Environment Methods
    # --------------------------------------------------------------------------------

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        """Advances the environment by one timestep."""
        action = self.action_convert(input_action)
        u, v = state.fluid_u, state.fluid_v

        # --- 1. Evolve the Fluid (if in chaotic mode) ---
        if self.is_chaotic:
            # Add a constant external force to drive the fluid, creating persistent currents
            force_u = jnp.zeros_like(u)
            force_v = jnp.zeros_like(v)
            # Example: A circular force field
            x_coords, y_coords = jnp.meshgrid(jnp.arange(self.fluid_grid_size), jnp.arange(self.fluid_grid_size))
            center_x, center_y = self.fluid_grid_size / 2, self.fluid_grid_size / 2
            dx, dy = x_coords - center_x, y_coords - center_y
            force_u = -dy * self.fluid_force * 1e-4
            force_v = dx * self.fluid_force * 1e-4

            u += self.fluid_dt * force_u
            v += self.fluid_dt * force_v

            # Standard fluid solver steps (Stable Fluids method)
            u = self._diffuse(u, self.fluid_viscosity, self.fluid_dt, self.fluid_grid_size, self.fluid_iterations)
            v = self._diffuse(v, self.fluid_viscosity, self.fluid_dt, self.fluid_grid_size, self.fluid_iterations)
            u, v = self._project(u, v, self.fluid_iterations)

            u = self._advect(u, u, v, self.fluid_dt, self.fluid_grid_size)
            v = self._advect(v, u, v, self.fluid_dt, self.fluid_grid_size)
            u, v = self._project(u, v, self.fluid_iterations)

        # --- 2. Get Current at Boat's Position ---
        current = self.current_func(state, key)

        # --- 3. Update Boat Position ---
        x_hat = state.x + action[0] + current[0]
        y_hat = state.y + action[1] + current[1]

        # --- 4. Create New State ---
        new_state = EnvState(x=x_hat, y=y_hat, time=state.time + 1, key=key, fluid_u=u, fluid_v=v)
        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {})

    def current_func(self, state: EnvState, key: chex.PRNGKey) -> chex.Array:
        """Calculates the current at the boat's position."""
        if not self.is_chaotic:
            # Fallback to a simple sinusoidal current if not in chaotic mode
            return jnp.array((-0.2 + 0.1 * jnp.sin(state.time * 0.1), 0.2))

        # Convert boat's world coordinates to fluid grid coordinates
        grid_x = state.x / self.width * self.fluid_grid_size
        grid_y = state.y / self.length * self.fluid_grid_size

        # Create coordinate array for interpolation
        coords = jnp.array([[grid_y], [grid_x]])

        # Bilinearly interpolate the velocity from the fluid grid
        u_interpolated = jax.scipy.ndimage.map_coordinates(state.fluid_u, coords, order=1, mode='wrap')[0]
        v_interpolated = jax.scipy.ndimage.map_coordinates(state.fluid_v, coords, order=1, mode='wrap')[0]

        # The solver velocity is in units of (grid_cells / fluid_dt).
        # We return this as a displacement vector for the agent's timestep.
        # A scaling factor is used to make the current's effect noticeable.
        return jnp.array([u_interpolated, v_interpolated]) * self.fluid_force

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Resets the environment to an initial state."""
        key, u_key, v_key = jrandom.split(key, 3)

        # Initial fluid state: start with random noise to create chaos
        if self.is_chaotic:
            u_init = jrandom.normal(u_key, (self.fluid_grid_size, self.fluid_grid_size)) * 0.1
            v_init = jrandom.normal(v_key, (self.fluid_grid_size, self.fluid_grid_size)) * 0.1
            u_init, v_init = self._project(u_init, v_init, self.fluid_iterations)
        else:
            u_init = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))
            v_init = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))

        state = EnvState(x=jnp.zeros(()),
                         y=jnp.zeros(()),
                         time=0,
                         key=key,
                         fluid_u=u_init,
                         fluid_v=v_init)

        return self.get_obs(state), state

    def reward_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> chex.Array:
        """Calculates the reward, defined as the negative distance to the goal."""
        dist_to_goal = jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y)) - self.goal_state)
        return -dist_to_goal

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        """Clips the action to the maximum allowed value."""
        return jnp.clip(action, -self.max_action, self.max_action)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        """Returns the observable part of the state (boat's position)."""
        return jnp.array([state.x, state.y])

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        """Reconstructs a partial state from an observation (time and fluid are unknown)."""
        # Note: This cannot reconstruct the fluid state. Use with caution.
        dummy_fluid = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))
        return EnvState(x=obs[0], y=obs[1], time=-1, key=key, fluid_u=dummy_fluid, fluid_v=dummy_fluid)

    def is_done(self, state: EnvState) -> chex.Array:
        """Checks if the episode has ended."""
        # Check if out of bounds
        x_bounds = jnp.logical_or(state.x >= self.width, state.x < 0)
        y_bounds = jnp.logical_or(state.y >= self.length, state.y < 0)
        bounds = jnp.logical_or(x_bounds, y_bounds)

        # Check if goal is reached (within a small tolerance)
        goal_reached = jnp.linalg.norm(jnp.array((state.x, state.y)) - self.goal_state) < 0.5

        # Check if time horizon is reached
        timeout = state.time >= self.horizon

        return jnp.logical_or(jnp.logical_or(bounds, goal_reached), timeout)

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations/"):
        """Renders a trajectory to a GIF, visualizing the fluid dynamics."""
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        # Setup fine grid for colormesh and coarse grid for quiver plot
        fine_res = 50
        x_fine = jnp.linspace(0, self.width, fine_res)
        y_fine = jnp.linspace(0, self.length, fine_res)
        X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine)

        coarse_res = 15
        x_coarse = jnp.linspace(0, self.width, coarse_res)
        y_coarse = jnp.linspace(0, self.length, coarse_res)
        X_coarse, Y_coarse = jnp.meshgrid(x_coarse, y_coarse)

        # Helper to get current at a point using a specific fluid grid
        def get_current_at_point(x, y, u_grid, v_grid):
            grid_x = x / self.width * self.fluid_grid_size
            grid_y = y / self.length * self.fluid_grid_size
            coords = jnp.array([[grid_y], [grid_x]])
            u_interp = jax.scipy.ndimage.map_coordinates(u_grid, coords, order=1, mode='wrap')[0]
            v_interp = jax.scipy.ndimage.map_coordinates(v_grid, coords, order=1, mode='wrap')[0]
            return jnp.array([u_interp, v_interp])

        vmap_current_spatial = jax.vmap(jax.vmap(get_current_at_point, in_axes=(0, None, None, None)),
                                        in_axes=(None, 0, None, None))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(self.name)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal', adjustable='box')
        ax.plot(self.goal_state[0], self.goal_state[1], marker='*', markersize=15, color="gold", label="Goal State",
                zorder=4)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

        # Get initial fluid state for the first frame
        initial_u_grid, initial_v_grid = trajectory_state.fluid_u[0], trajectory_state.fluid_v[0]
        initial_current_fine = vmap_current_spatial(x_fine, y_fine, initial_u_grid, initial_v_grid)
        initial_mag_fine = jnp.linalg.norm(initial_current_fine, axis=-1)
        initial_current_coarse = vmap_current_spatial(x_coarse, y_coarse, initial_u_grid, initial_v_grid)
        initial_U, initial_V = initial_current_coarse[:, :, 0], initial_current_coarse[:, :, 1]

        # Setup animation elements
        line, = ax.plot([], [], 'r-', lw=2, label='Agent Trail', zorder=3)
        dot, = ax.plot([], [], color="magenta", marker="o", markersize=10, label='Current State', zorder=5)
        pcm = ax.pcolormesh(X_fine, Y_fine, initial_mag_fine, cmap='viridis', shading='auto', zorder=1, alpha=0.7)
        arrow = ax.quiver(X_coarse, Y_coarse, initial_U, initial_V, color='white', angles='xy', scale_units='xy',
                          scale=1.0, width=0.004, zorder=2)
        ax.legend(loc='upper left')
        fig.colorbar(pcm, ax=ax, shrink=0.8, label='Current Magnitude')
        agent_path_x, agent_path_y = [], []

        def update(frame):
            if frame == 0:
                agent_path_x.clear()
                agent_path_y.clear()

            agent_path_x.append(trajectory_state.x[frame])
            agent_path_y.append(trajectory_state.y[frame])
            line.set_data(agent_path_x, agent_path_y)
            dot.set_data([trajectory_state.x[frame]], [trajectory_state.y[frame]])

            # Get the fluid grid for the current frame
            frame_u_grid, frame_v_grid = trajectory_state.fluid_u[frame], trajectory_state.fluid_v[frame]

            current_vectors_fine = vmap_current_spatial(x_fine, y_fine, frame_u_grid, frame_v_grid)
            magnitude_fine = jnp.linalg.norm(current_vectors_fine, axis=-1)
            current_vectors_coarse = vmap_current_spatial(x_coarse, y_coarse, frame_u_grid, frame_v_grid)
            U_coarse, V_coarse = current_vectors_coarse[:, :, 0], current_vectors_coarse[:, :, 1]

            pcm.set_array(magnitude_fine.ravel())
            arrow.set_UVC(U_coarse, V_coarse)
            return line, dot, pcm, arrow

        anim = animation.FuncAnimation(fig, update, frames=len(trajectory_state.time), interval=100, blit=True)
        anim.save(f"{file_path}_{self.name}.gif", writer='imagemagick')
        plt.close()

    @property
    def name(self) -> str:
        return "BoatInCurrent-v0-Chaotic"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_action, self.max_action, shape=(2,))

    def observation_space(self) -> spaces.Box:
        low = jnp.array([0, 0])
        high = jnp.array([self.width, self.length])
        return spaces.Box(low, high, (2,))


import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.ndimage
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex
from functools import partial


@struct.dataclass
class EnvState(base_env.EnvState):
    """
    Represents the state of the environment.

    Attributes:
        x (chex.Array): The x-coordinate of the boat.
        y (chex.Array): The y-coordinate of the boat.
        time (int): The current timestep.
        key (chex.PRNGKey): JAX random key.
        fluid_u (chex.Array): The u-component (x-velocity) of the fluid grid.
        fluid_v (chex.Array): The v-component (y-velocity) of the fluid grid.
    """
    x: chex.Array
    y: chex.Array
    time: int
    key: chex.PRNGKey
    # New state for the fluid grid
    fluid_u: chex.Array
    fluid_v: chex.Array


class BoatInCurrentCSCA(base_env.BaseEnvironment):
    """
    A 2D environment simulating a boat in a current.
    The current can be modeled in several ways, including a chaotic fluid solver.
    """

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # Environment dimensions
        self.length: float = 15.0
        self.width: float = 15.0

        # Goal state
        self.goal_state = jnp.array((self.length, self.width))

        # Action and horizon limits
        self.max_action: float = 1.0
        self.horizon: int = 200

        # --- Fluid Dynamics Parameters for Chaotic Current ---
        self.is_chaotic: bool = True  # Set to True to use the fluid solver
        self.fluid_grid_size: int = 64  # Resolution of the fluid grid (NxN)
        self.fluid_dt: float = 0.2  # Timestep for the fluid simulation
        self.fluid_viscosity: float = 1e-7  # Viscosity of the fluid
        self.fluid_iterations: int = 20  # Number of iterations for the linear solver
        self.fluid_force: float = 5.0  # Strength of the external force driving the fluidf
        self.fluid_damping: float = 0.999  # Damping factor to prevent velocity explosion
        self.current_scaling_factor: float = 0.2  # Scales the effect of the current on the boat

    # --------------------------------------------------------------------------------
    # JIT-compiled static methods for the fluid solver
    # Based on "Stable Fluids" by Jos Stam
    # --------------------------------------------------------------------------------

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _linear_solve(x: chex.Array, b: chex.Array, a: float, c: float, num_iterations: int) -> chex.Array:
        """
        Solves a linear system using Jacobi iteration with periodic boundary conditions.
        Used for diffusion and pressure projection.
        """

        def body_fun(_, val):
            x_prev = val
            # Neighbors are found using jnp.roll for periodic boundaries
            neighbors = (jnp.roll(x_prev, 1, axis=0) + jnp.roll(x_prev, -1, axis=0) +
                         jnp.roll(x_prev, 1, axis=1) + jnp.roll(x_prev, -1, axis=1))
            x_new = (b + a * neighbors) / c
            return x_new

        return jax.lax.fori_loop(0, num_iterations, body_fun, x)

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def _diffuse(field: chex.Array, viscosity: float, dt: float, grid_size: int, num_iterations: int) -> chex.Array:
        """Applies fluid diffusion (viscosity)."""
        a = dt * viscosity * grid_size * grid_size
        return BoatInCurrentCSCA._linear_solve(field, field, a, 1 + 4 * a, num_iterations)

    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4))
    def _advect(field: chex.Array, u: chex.Array, v: chex.Array, dt: float, grid_size: int) -> chex.Array:
        """Moves a quantity 'field' through the velocity field (u, v)."""
        N = grid_size
        x_coords, y_coords = jnp.meshgrid(jnp.arange(N), jnp.arange(N))

        # Trace back in time to find the source of the fluid
        back_x = x_coords - (dt * u * N)
        back_y = y_coords - (dt * v * N)

        coords = jnp.stack([back_y, back_x], axis=0)

        # Sample from the original field at the back-traced coordinates
        return jax.scipy.ndimage.map_coordinates(field, coords, order=1, mode='wrap')

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _project(u: chex.Array, v: chex.Array, num_iterations: int) -> Tuple[chex.Array, chex.Array]:
        """Enforces fluid incompressibility."""
        # Calculate divergence using central differences
        div = -0.5 * (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1) +
                      jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0))

        # Solve Poisson equation for pressure
        p = jnp.zeros_like(div)
        p = BoatInCurrentCSCA._linear_solve(p, -div, 1.0, 4.0, num_iterations)

        # Subtract the pressure gradient from the velocity field
        u_new = u - 0.5 * (jnp.roll(p, -1, axis=1) - jnp.roll(p, 1, axis=1))
        v_new = v - 0.5 * (jnp.roll(p, -1, axis=0) - jnp.roll(p, 1, axis=0))

        return u_new, v_new

    # --------------------------------------------------------------------------------
    # Core Environment Methods
    # --------------------------------------------------------------------------------

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        """Advances the environment by one timestep."""
        action = self.action_convert(input_action)
        u, v = state.fluid_u, state.fluid_v

        # --- 1. Evolve the Fluid (if in chaotic mode) ---
        if self.is_chaotic:
            # Add a constant external force to drive the fluid, creating persistent currents
            force_u = jnp.zeros_like(u)
            force_v = jnp.zeros_like(v)
            # Example: A circular force field
            x_coords, y_coords = jnp.meshgrid(jnp.arange(self.fluid_grid_size), jnp.arange(self.fluid_grid_size))
            center_x, center_y = self.fluid_grid_size / 2, self.fluid_grid_size / 2
            dx, dy = x_coords - center_x, y_coords - center_y
            force_u = -dy * self.fluid_force * 1e-4
            force_v = dx * self.fluid_force * 1e-4

            u += self.fluid_dt * force_u
            v += self.fluid_dt * force_v

            # Standard fluid solver steps (Stable Fluids method)
            u = self._diffuse(u, self.fluid_viscosity, self.fluid_dt, self.fluid_grid_size, self.fluid_iterations)
            v = self._diffuse(v, self.fluid_viscosity, self.fluid_dt, self.fluid_grid_size, self.fluid_iterations)
            u, v = self._project(u, v, self.fluid_iterations)

            u = self._advect(u, u, v, self.fluid_dt, self.fluid_grid_size)
            v = self._advect(v, u, v, self.fluid_dt, self.fluid_grid_size)
            u, v = self._project(u, v, self.fluid_iterations)

            # --- Apply Damping ---
            # This is the crucial step to prevent the fluid velocity from growing indefinitely.
            # It simulates energy loss due to factors not explicitly modeled, like friction.
            u *= self.fluid_damping
            v *= self.fluid_damping

        # --- 2. Get Current at Boat's Position ---
        # Note: we pass the *newly updated* fluid state to the current function
        current_state_for_boat = EnvState(x=state.x, y=state.y, time=state.time, key=key, fluid_u=u, fluid_v=v)
        current = self.current_func(current_state_for_boat, key)

        # --- 3. Update Boat Position ---
        x_hat = state.x + action[0] + current[0]
        y_hat = state.y + action[1] + current[1]

        # --- 4. Create New State ---
        new_state = EnvState(x=x_hat, y=y_hat, time=state.time + 1, key=key, fluid_u=u, fluid_v=v)
        reward = self.reward_function(input_action, state, new_state, key)

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                self.is_done(new_state),
                {})

    def current_func(self, state: EnvState, key: chex.PRNGKey) -> chex.Array:
        """Calculates the current at the boat's position."""
        if not self.is_chaotic:
            # Fallback to a simple sinusoidal current if not in chaotic mode
            return jnp.array((-0.2 + 0.1 * jnp.sin(state.time * 0.1), 0.2))

        # Convert boat's world coordinates to fluid grid coordinates
        grid_x = state.x / self.width * self.fluid_grid_size
        grid_y = state.y / self.length * self.fluid_grid_size

        # Create coordinate array for interpolation
        coords = jnp.array([[grid_y], [grid_x]])

        # Bilinearly interpolate the velocity from the fluid grid
        u_interpolated = jax.scipy.ndimage.map_coordinates(state.fluid_u, coords, order=1, mode='wrap')[0]
        v_interpolated = jax.scipy.ndimage.map_coordinates(state.fluid_v, coords, order=1, mode='wrap')[0]

        # The solver velocity is in units of (grid_cells / fluid_dt).
        # We scale this to get a reasonable displacement for the agent's timestep.
        return jnp.array([u_interpolated, v_interpolated]) * self.current_scaling_factor

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Resets the environment to an initial state."""
        key, u_key, v_key = jrandom.split(key, 3)

        # Initial fluid state: start with random noise to create chaos
        if self.is_chaotic:
            u_init = jrandom.normal(u_key, (self.fluid_grid_size, self.fluid_grid_size)) * 0.1
            v_init = jrandom.normal(v_key, (self.fluid_grid_size, self.fluid_grid_size)) * 0.1
            u_init, v_init = self._project(u_init, v_init, self.fluid_iterations)
        else:
            u_init = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))
            v_init = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))

        state = EnvState(x=jnp.zeros(()),
                         y=jnp.zeros(()),
                         time=0,
                         key=key,
                         fluid_u=u_init,
                         fluid_v=v_init)

        return self.get_obs(state), state

    def reward_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> chex.Array:
        """Calculates the reward, defined as the negative distance to the goal."""
        dist_to_goal = jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y)) - self.goal_state)
        return -dist_to_goal

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        """Clips the action to the maximum allowed value."""
        return jnp.clip(action, -self.max_action, self.max_action)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        """Returns the observable part of the state (boat's position)."""
        return jnp.array([state.x, state.y])

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        """Reconstructs a partial state from an observation (time and fluid are unknown)."""
        # Note: This cannot reconstruct the fluid state. Use with caution.
        dummy_fluid = jnp.zeros((self.fluid_grid_size, self.fluid_grid_size))
        return EnvState(x=obs[0], y=obs[1], time=-1, key=key, fluid_u=dummy_fluid, fluid_v=dummy_fluid)

    def is_done(self, state: EnvState) -> chex.Array:
        """Checks if the episode has ended."""
        # Check if out of bounds
        x_bounds = jnp.logical_or(state.x >= self.width, state.x < 0)
        y_bounds = jnp.logical_or(state.y >= self.length, state.y < 0)
        bounds = jnp.logical_or(x_bounds, y_bounds)

        # Check if goal is reached (within a small tolerance)
        goal_reached = jnp.linalg.norm(jnp.array((state.x, state.y)) - self.goal_state) < 0.5

        # Check if time horizon is reached
        timeout = state.time >= self.horizon

        return jnp.logical_or(jnp.logical_or(bounds, goal_reached), timeout)

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations/"):
        """Renders a trajectory to a GIF, visualizing the fluid dynamics."""
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        # Setup fine grid for colormesh and coarse grid for quiver plot
        fine_res = 50
        x_fine = jnp.linspace(0, self.width, fine_res)
        y_fine = jnp.linspace(0, self.length, fine_res)
        X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine)

        coarse_res = 15
        x_coarse = jnp.linspace(0, self.width, coarse_res)
        y_coarse = jnp.linspace(0, self.length, coarse_res)
        X_coarse, Y_coarse = jnp.meshgrid(x_coarse, y_coarse)

        # Helper to get current at a point using a specific fluid grid
        def get_current_at_point(x, y, u_grid, v_grid):
            grid_x = x / self.width * self.fluid_grid_size
            grid_y = y / self.length * self.fluid_grid_size
            coords = jnp.array([[grid_y], [grid_x]])
            u_interp = jax.scipy.ndimage.map_coordinates(u_grid, coords, order=1, mode='wrap')[0]
            v_interp = jax.scipy.ndimage.map_coordinates(v_grid, coords, order=1, mode='wrap')[0]
            # We scale here for visualization purposes
            return jnp.array([u_interp, v_interp]) * self.current_scaling_factor

        vmap_current_spatial = jax.vmap(jax.vmap(get_current_at_point, in_axes=(0, None, None, None)),
                                        in_axes=(None, 0, None, None))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(self.name)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal', adjustable='box')
        ax.plot(self.goal_state[0], self.goal_state[1], marker='*', markersize=15, color="gold", label="Goal State",
                zorder=4)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

        # Get initial fluid state for the first frame
        initial_u_grid, initial_v_grid = trajectory_state.fluid_u[0], trajectory_state.fluid_v[0]
        initial_current_fine = vmap_current_spatial(x_fine, y_fine, initial_u_grid, initial_v_grid)
        initial_mag_fine = jnp.linalg.norm(initial_current_fine, axis=-1)
        initial_current_coarse = vmap_current_spatial(x_coarse, y_coarse, initial_u_grid, initial_v_grid)
        initial_U, initial_V = initial_current_coarse[:, :, 0], initial_current_coarse[:, :, 1]

        # Setup animation elements
        line, = ax.plot([], [], 'r-', lw=2, label='Agent Trail', zorder=3)
        dot, = ax.plot([], [], color="magenta", marker="o", markersize=10, label='Current State', zorder=5)
        pcm = ax.pcolormesh(X_fine, Y_fine, initial_mag_fine, cmap='viridis', shading='auto', zorder=1, alpha=0.7,
                            vmin=0, vmax=0.5)
        arrow = ax.quiver(X_coarse, Y_coarse, initial_U, initial_V, color='white', angles='xy', scale_units='xy',
                          scale=1.0, width=0.004, zorder=2)
        ax.legend(loc='upper left')
        fig.colorbar(pcm, ax=ax, shrink=0.8, label='Current Magnitude')
        agent_path_x, agent_path_y = [], []

        def update(frame):
            if frame == 0:
                agent_path_x.clear()
                agent_path_y.clear()

            agent_path_x.append(trajectory_state.x[frame])
            agent_path_y.append(trajectory_state.y[frame])
            line.set_data(agent_path_x, agent_path_y)
            dot.set_data([trajectory_state.x[frame]], [trajectory_state.y[frame]])

            # Get the fluid grid for the current frame
            frame_u_grid, frame_v_grid = trajectory_state.fluid_u[frame], trajectory_state.fluid_v[frame]

            current_vectors_fine = vmap_current_spatial(x_fine, y_fine, frame_u_grid, frame_v_grid)
            magnitude_fine = jnp.linalg.norm(current_vectors_fine, axis=-1)
            current_vectors_coarse = vmap_current_spatial(x_coarse, y_coarse, frame_u_grid, frame_v_grid)
            U_coarse, V_coarse = current_vectors_coarse[:, :, 0], current_vectors_coarse[:, :, 1]

            pcm.set_array(magnitude_fine.ravel())
            arrow.set_UVC(U_coarse, V_coarse)
            return line, dot, pcm, arrow

        anim = animation.FuncAnimation(fig, update, frames=len(trajectory_state.time), interval=100, blit=True)
        anim.save(f"{file_path}_{self.name}.gif", writer='imagemagick')
        plt.close()

    @property
    def name(self) -> str:
        return "BoatInCurrent-v0-Chaotic"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_action, self.max_action, shape=(2,))

    def observation_space(self) -> spaces.Box:
        low = jnp.array([0, 0])
        high = jnp.array([self.width, self.length])
        return spaces.Box(low, high, (2,))


import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.ndimage
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
from typing import Any, Dict, Tuple, Union
import chex
from functools import partial


@struct.dataclass
class EnvState(base_env.EnvState):
    """
    Represents the state of the environment.

    Attributes:
        x (chex.Array): The x-coordinate of the boat.
        y (chex.Array): The y-coordinate of the boat.
        time (int): The current timestep.
        key (chex.PRNGKey): JAX random key.
        fluid_u (chex.Array): The u-component (x-velocity) of the fluid grid in world units/sec.
        fluid_v (chex.Array): The v-component (y-velocity) of the fluid grid in world units/sec.
    """
    x: chex.Array
    y: chex.Array
    time: int
    key: chex.PRNGKey
    fluid_u: chex.Array
    fluid_v: chex.Array


class BoatInCurrentCSCA(base_env.BaseEnvironment):
    """
    A 2D environment simulating a boat in a current.
    The fluid solver uses a grid of SQUARE cells, which can form a RECTANGULAR overall grid.
    """

    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        # Environment dimensions
        self.length: float = 20.0  # Example of a rectangular world
        self.width: float = 10.0

        # Goal state
        self.goal_state = jnp.array((self.width, self.length))

        # Action and horizon limits
        self.max_action: float = 1.0
        self.horizon: int = 200
        self.agent_dt: float = 1.0  # Timestep for a single agent action

        # --- Fluid Dynamics Parameters for Chaotic Current ---
        self.is_chaotic: bool = True
        self.cell_size: float = 0.5  # The side length of each SQUARE grid cell in world units.
        self.fluid_dt: float = 0.1  # Timestep for the fluid simulation
        self.fluid_viscosity: float = 1e-6
        self.fluid_iterations: int = 20
        self.fluid_force: float = 5.0
        self.fluid_damping: float = 0.999

        # --- Rectangular Grid Setup from Square Cells ---
        self.fluid_grid_width = int(self.width / self.cell_size)
        self.fluid_grid_height = int(self.length / self.cell_size)

        # For clarity, dx and dy are the same, representing the size of our square cells.
        self.dx = self.cell_size
        self.dy = self.cell_size

    # --------------------------------------------------------------------------------
    # JIT-compiled static methods for the fluid solver (for square cells on a rectangular grid)
    # --------------------------------------------------------------------------------

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _linear_solve(x: chex.Array, b: chex.Array, a: float, c: float, num_iterations: int) -> chex.Array:
        """
        Solves a linear system using Jacobi iteration. Simplified for square cells (isotropic).
        """

        def body_fun(_, val):
            x_prev = val
            neighbors = (jnp.roll(x_prev, 1, axis=0) + jnp.roll(x_prev, -1, axis=0) +
                         jnp.roll(x_prev, 1, axis=1) + jnp.roll(x_prev, -1, axis=1))
            x_new = (b + a * neighbors) / c
            return x_new

        return jax.lax.fori_loop(0, num_iterations, body_fun, x)

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def _diffuse(field, viscosity, dt, cell_size, num_iterations):
        """Applies fluid diffusion. Simplified for square cells."""
        a = dt * viscosity / (cell_size * cell_size)
        return BoatInCurrentCSCA._linear_solve(field, field, a, 1 + 4 * a, num_iterations)

    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4, 5))
    def _advect(field, u, v, dt, cell_size, grid_shape):
        """Moves a quantity 'field' through the velocity field (u, v) in world units."""
        grid_height, grid_width = grid_shape
        y_coords_grid, x_coords_grid = jnp.mgrid[0:grid_height, 0:grid_width]

        # Convert grid coordinates to world coordinates
        x_coords_world = x_coords_grid * cell_size
        y_coords_world = y_coords_grid * cell_size

        # Trace back in time in world space
        back_x_world = x_coords_world - dt * u
        back_y_world = y_coords_world - dt * v

        # Convert back-traced world coordinates to grid coordinates for sampling
        back_x_grid = back_x_world / cell_size
        back_y_grid = back_y_world / cell_size

        coords = jnp.stack([back_y_grid, back_x_grid], axis=0)
        return jax.scipy.ndimage.map_coordinates(field, coords, order=1, mode='wrap')

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def _project(u, v, cell_size, num_iterations):
        """Enforces fluid incompressibility. Simplified for square cells."""
        # Discretized divergence: du/dx + dv/dy
        div = -0.5 / cell_size * ((jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) +
                                  (jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0)))

        # Solve Poisson equation: ∇²p = div. For square cells, this simplifies.
        p = jnp.zeros_like(div)
        p = BoatInCurrentCSCA._linear_solve(p, div, 1.0, 4.0, num_iterations)

        # Subtract pressure gradient: V' = V - ∇p
        u_new = u - 0.5 / cell_size * (jnp.roll(p, -1, axis=1) - jnp.roll(p, 1, axis=1))
        v_new = v - 0.5 / cell_size * (jnp.roll(p, -1, axis=0) - jnp.roll(p, 1, axis=0))

        return u_new, v_new

    # --------------------------------------------------------------------------------
    # Core Environment Methods
    # --------------------------------------------------------------------------------

    def step_env(self, input_action, state, key):
        action = self.action_convert(input_action)
        u, v = state.fluid_u, state.fluid_v

        if self.is_chaotic:
            # --- Step 1: Add Force (as an acceleration) ---
            y_coords, x_coords = jnp.mgrid[0:self.fluid_grid_height, 0:self.fluid_grid_width]
            center_y, center_x = self.fluid_grid_height / 2, self.fluid_grid_width / 2
            dx_grid, dy_grid = x_coords - center_x, y_coords - center_y

            force_u = -dy_grid * self.fluid_force * 1e-2
            force_v = dx_grid * self.fluid_force * 1e-2

            u += self.fluid_dt * force_u
            v += self.fluid_dt * force_v

            # --- FLUID SOLVER SEQUENCE for dt ---
            u = self._diffuse(u, self.fluid_viscosity, self.fluid_dt, self.cell_size, self.fluid_iterations)
            v = self._diffuse(v, self.fluid_viscosity, self.fluid_dt, self.cell_size, self.fluid_iterations)

            grid_shape = (self.fluid_grid_height, self.fluid_grid_width)
            u = self._advect(u, u, v, self.fluid_dt, self.cell_size, grid_shape)
            v = self._advect(v, u, v, self.fluid_dt, self.cell_size, grid_shape)

            u, v = self._project(u, v, self.cell_size, self.fluid_iterations)

            u *= self.fluid_damping
            v *= self.fluid_damping

        current = self.current_func(state._replace(fluid_u=u, fluid_v=v), key)
        x_hat = state.x + action[0] + current[0]
        y_hat = state.y + action[1] + current[1]

        new_state = state._replace(x=x_hat, y=y_hat, time=state.time + 1, key=key, fluid_u=u, fluid_v=v)
        reward = self.reward_function(input_action, state, new_state, key)

        return (self.get_obs(new_state), self.get_obs(new_state) - self.get_obs(state),
                new_state, reward, self.is_done(new_state), {})

    def current_func(self, state: EnvState, key: chex.PRNGKey) -> chex.Array:
        """Calculates the current displacement at the boat's position for one agent timestep."""
        if not self.is_chaotic:
            return jnp.array([-0.1, 0.1]) * self.agent_dt

        # Convert boat's world coordinates to fluid grid coordinates for sampling
        grid_x = state.x / self.cell_size
        grid_y = state.y / self.cell_size
        coords = jnp.array([[grid_y], [grid_x]])

        # Interpolate the velocity (world units/sec) from the fluid grid
        u_interpolated = jax.scipy.ndimage.map_coordinates(state.fluid_u, coords, order=1, mode='wrap')[0]
        v_interpolated = jax.scipy.ndimage.map_coordinates(state.fluid_v, coords, order=1, mode='wrap')[0]

        # Return displacement = velocity * agent timestep
        return jnp.array([u_interpolated, v_interpolated]) * self.agent_dt

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        key, u_key, v_key = jrandom.split(key, 3)
        grid_shape = (self.fluid_grid_height, self.fluid_grid_width)

        if self.is_chaotic:
            u_init = jrandom.normal(u_key, grid_shape) * 0.1
            v_init = jrandom.normal(v_key, grid_shape) * 0.1
            u_init, v_init = self._project(u_init, v_init, self.cell_size, self.fluid_iterations)
        else:
            u_init = jnp.zeros(grid_shape)
            v_init = jnp.zeros(grid_shape)

        state = EnvState(x=jnp.zeros(()), y=jnp.zeros(()), time=0, key=key, fluid_u=u_init, fluid_v=v_init)
        return self.get_obs(state), state

    def reward_function(self, input_action_t, state_t, state_tp1, key=None):
        dist_to_goal = jnp.linalg.norm(jnp.array((state_tp1.x, state_tp1.y)) - self.goal_state)
        return -dist_to_goal

    def action_convert(self, action):
        return jnp.clip(action, -self.max_action, self.max_action)

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        return jnp.array([state.x, state.y])

    def get_state(self, obs: chex.Array, key: chex.PRNGKey = None) -> EnvState:
        dummy_fluid_shape = (self.fluid_grid_height, self.fluid_grid_width)
        dummy_fluid = jnp.zeros(dummy_fluid_shape)
        return EnvState(x=obs[0], y=obs[1], time=-1, key=key, fluid_u=dummy_fluid, fluid_v=dummy_fluid)

    def is_done(self, state: EnvState) -> chex.Array:
        x_bounds = (state.x >= self.width) | (state.x < 0)
        y_bounds = (state.y >= self.length) | (state.y < 0)
        goal_reached = jnp.linalg.norm(jnp.array((state.x, state.y)) - self.goal_state) < 0.5
        timeout = state.time >= self.horizon
        return x_bounds | y_bounds | goal_reached | timeout

    def render_traj(self, trajectory_state: EnvState, file_path: str = "../animations/"):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        # Adjust plot size to match world aspect ratio
        fig_width = 8
        fig_height = fig_width * (self.length / self.width)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.set_title(self.name)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal', adjustable='box')
        ax.plot(self.goal_state[0], self.goal_state[1], marker='*', markersize=15, color="gold", label="Goal State",
                zorder=4)

        # Setup grids for visualization
        fine_res_x = self.fluid_grid_width * 2
        fine_res_y = self.fluid_grid_height * 2
        x_fine = jnp.linspace(0, self.width, fine_res_x)
        y_fine = jnp.linspace(0, self.length, fine_res_y)
        X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine)

        coarse_res_x = 20
        coarse_res_y = int(coarse_res_x * (self.length / self.width))
        x_coarse = jnp.linspace(0, self.width, coarse_res_x)
        y_coarse = jnp.linspace(0, self.length, coarse_res_y)
        X_coarse, Y_coarse = jnp.meshgrid(x_coarse, y_coarse)

        def get_velocity_at_point(x, y, u_grid, v_grid):
            grid_x = x / self.cell_size
            grid_y = y / self.cell_size
            coords = jnp.array([[grid_y], [grid_x]])
            u_interp = jax.scipy.ndimage.map_coordinates(u_grid, coords, order=1, mode='wrap')[0]
            v_interp = jax.scipy.ndimage.map_coordinates(v_grid, coords, order=1, mode='wrap')[0]
            return jnp.array([u_interp, v_interp])

        vmap_current_spatial = jax.vmap(jax.vmap(get_velocity_at_point, in_axes=(0, None, None, None)),
                                        in_axes=(None, 0, None, None))

        initial_u, initial_v = trajectory_state.fluid_u[0], trajectory_state.fluid_v[0]
        initial_vel_fine = vmap_current_spatial(x_fine, y_fine, initial_u, initial_v)
        initial_mag_fine = jnp.linalg.norm(initial_vel_fine, axis=-1)
        initial_vel_coarse = vmap_current_spatial(x_coarse, y_coarse, initial_u, initial_v)

        line, = ax.plot([], [], 'r-', lw=2, label='Agent Trail', zorder=3)
        dot, = ax.plot([], [], color="magenta", marker="o", markersize=10, label='Current State', zorder=5)
        pcm = ax.pcolormesh(X_fine, Y_fine, initial_mag_fine, cmap='viridis', shading='auto', zorder=1, alpha=0.7,
                            vmin=0, vmax=1.0)
        arrow = ax.quiver(X_coarse, Y_coarse, initial_vel_coarse[:, :, 0], initial_vel_coarse[:, :, 1], color='white',
                          angles='xy', scale_units='xy', scale=5.0, width=0.004, zorder=2)

        ax.legend(loc='upper left')
        fig.colorbar(pcm, ax=ax, shrink=0.8, label='Current Magnitude (m/s)')
        agent_path_x, agent_path_y = [], []

        def update(frame):
            if frame == 0: agent_path_x.clear(); agent_path_y.clear()
            agent_path_x.append(trajectory_state.x[frame]);
            agent_path_y.append(trajectory_state.y[frame])
            line.set_data(agent_path_x, agent_path_y)
            dot.set_data([trajectory_state.x[frame]], [trajectory_state.y[frame]])

            u_grid, v_grid = trajectory_state.fluid_u[frame], trajectory_state.fluid_v[frame]
            vel_fine = vmap_current_spatial(x_fine, y_fine, u_grid, v_grid)
            magnitude_fine = jnp.linalg.norm(vel_fine, axis=-1)
            vel_coarse = vmap_current_spatial(x_coarse, y_coarse, u_grid, v_grid)

            pcm.set_array(magnitude_fine.ravel())
            arrow.set_UVC(vel_coarse[:, :, 0], vel_coarse[:, :, 1])
            return line, dot, pcm, arrow

        anim = animation.FuncAnimation(fig, update, frames=len(trajectory_state.time), interval=100, blit=True)
        anim.save(f"{file_path}_{self.name}.gif", writer='imagemagick')
        plt.close()

    @property
    def name(self) -> str:
        return f"BoatInCurrent-{self.width}x{self.length}-Chaotic"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_action, self.max_action, shape=(2,))

    def observation_space(self) -> spaces.Box:
        low = jnp.array([0, 0])
        high = jnp.array([self.width, self.length])
        return spaces.Box(low, high, (2,))

