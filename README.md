# BifurcaGym

Chaotic transition dynamics are found in many natural processes such as fluids, weather, financial systems, ecology, and 
even multi-agent systems.

In many of these scenarios we want to learn behavioural policies towards a desired objective, such as in fluid control 
(e.g. fluid mixing, aerofoil optimisation), financial or ecosystem control (a stable desired equilibrium), weather model 
parameter optimisation, autonomous vehicle control under wind/current effects (e.g. underwater drones or sailboats).

However, chaos has a huge impact on Reinforcement Learning or Optimal Control methods as we look to optimise for long 
horizon sequential tasks; under chaos trajectories exponentially diverge beyond Lyapunov time.

Due to chaotic systems having sensitivity to initial conditions, it can be hard to ensure robustness and accuracy in 
prediction and control in these types of environments.

BifurcaGym is a collection of environments that experience chaos in their transition dynamics, so that practitioners and 
theorists can understand the impact chaos has on their algorithm design. 

BifurcaGym is written in Jax to gain the benefits of the enhanced Autodiff and acceleration potential, not only for the 
non-linear environment dynamics (which many require extensive CFD/FEA/FEM solvers) but also in the reduction of 
overheads between the environment and the agent for Reinforcement Learning/Optimal Control. 

## The Framework

Since both Model-Based and Model-Free methods are typically used for these types of environments, we ensure that 
BifurcaGym environments are set up to provide the correct outputs and wrappers for both. For example outputs also 
contain delta obs (the change in observation between steps) as this is amenable for Model-Based setups. There is also a 
function called get_delta_obs that ensures correct periodicity is applied for certain environments if you are using a 
dynamics model to predict the transition change rather than just the next observation.

Environments can come in various discrete/continuous state and action space combinations, which can be easily selected 
when loading the environment. Wrappers are made for normalising state and action spaces as well as auto-resetting the 
environment to enable easy Jax based environment rollouts using jax.lax.scan. 

We also easily allow a generative step so that observations can be fed into the env, again useful for some Model-Based 
RL setups that utilise generative dynamics models.
Observations that are fed in must match the scale (aka if normalised or not) compared to the environment output.

Further, each env must have a defined reward function that again is amenable to Model-Based RL experiments where focus 
is placed on learning a transition function. We combine the reward and done function as a separate entity for faster
computation as well as for using the reward function if we do not look to learn it when using Model-Based methods.


## Installation

BifurcaGym uses [PDM](https://pdm-project.org) and Python 3.11 or 3.12.

```bash
pdm install                   # CPU
pdm install -G cuda           # NVIDIA GPU (CUDA 12)
pdm install -G baselines      # adds optax, distrax and wandb for the PPO baseline
pdm install -G hydrogym       # adds HydroGym for the HydroGym-* fluid control envs
pdm install -G mujoco         # adds MuJoCo and MJX for the Hopper and Walker2d locomotion envs
```

## Quick Start

```python
import jax.random as jrandom
import bifurcagym

env, env_params = bifurcagym.make("Lorenz63-v0", cont_state=True, cont_action=True, normalised=True, autoreset=True)

key = jrandom.key(42)
key, reset_key = jrandom.split(key)
obs, state = env.reset(env_params, reset_key)

key, action_key, step_key = jrandom.split(key, 3)
action = env.action_space(env_params).sample(action_key)
obs, delta_obs, state, reward, done, info = env.step(action, state, env_params, step_key)

# Model-Based RL: step from an observation instead of the hidden state, and query the reward function directly
obs_gen, delta_obs_gen, state_gen, reward_gen, done_gen, _ = env.generative_step(action, obs, env_params, step_key)
```

Because `env_params` is passed explicitly, environments can be `jax.vmap`-ed over both keys and parameters, for
example to sweep a bifurcation parameter or apply non-stationary changes.

### Precision

`make` runs in 64-bit by default. Pass `x64=False` for 32-bit, which is available for Pendulum, CartPole, WetChicken,
AYS-IAM, FluidicPinball, HydroGym-Kolmogorov, Hopper and Walker2d. The chaotic systems need float64, so `make` raises
an error if they are requested in 32-bit. This sets JAX's process wide `jax_enable_x64` flag, so every env made in a
process has to use the same precision, and arrays created before the first `make` follow the flag at that time.

## Environments

`C` = continuous, `D` = discrete, `D/C` = both are available through `cont_state` / `cont_action` in `make`.
`partial` marks environments where the observation is a partial view of the state, so `generative_step` is not
available (it is also unavailable for discrete state versions, as observations are projections of the state).

| Environment | ID | State Space | Action Space | Obs Dim | Action Dim | Working | Reference |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--|
| Acrobot | `Acrobot-v0` | C | D/C | 6 | 1 | Y | [Code](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/acrobot.py); [Paper](http://users.cms.caltech.edu/~murray/preprints/erl-M91-46.pdf) |
| Cartpole | `CartPole-v0` | C | D/C | 4 | 1 | Y | [Code](https://github.com/fusion-ml/trajectory-information-rl); [Paper](https://ieeexplore.ieee.org/document/6313077) |
| N Cartpole | `NCartPole-v0` | C | D/C | 6 | 1 | N | [Info](https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html) |
| N Pendulum | `NPendulum-v0` | C | D/C | 4 | 1 | N |  |
| Pendubot | `Pendubot-v0` | C | D/C | 6 | 1 | Y | [Paper](https://link.springer.com/chapter/10.1007/BFb0015081); [Info](https://underactuated.mit.edu/acrobot.html) |
| Pendulum | `Pendulum-v0` | C | D/C | 2 | 1 | Y | [Code](https://github.com/fusion-ml/trajectory-information-rl) |
| Wet Chicken | `WetChicken-v0` | D/C | D/C | 2 | 2 | Y | [Code](https://github.com/LAVA-LAB/improved_spi/blob/main/wetChicken.py); [Paper](https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Publikationen/conferences_int/2009/Hans-ICANN-2009.pdf) |
| Arnold–Beltrami–Childress Flow | `ABCFlow-v0` | C | D/C | 3 | 3 | Y | [Info](https://en.wikipedia.org/wiki/Arnold%E2%80%93Beltrami%E2%80%93Childress_flow) |
| Bickley Jet Flow | `BickleyJetFlow-v0` | C | D/C | 3 | 2 | Y | [Info](https://en.wikipedia.org/wiki/Bickley_jet) |
| Chua | `Chua-v0` | C | D/C | 3 | 2 | Y | [Info](http://www.scholarpedia.org/article/Chua_circuit) |
| Double Gyre Flow | `DoubleGyreFlow-v0` | C | D/C | 3 | 2 | Y | [Paper](https://www.nature.com/articles/s41467-021-27015-y.pdf); [Info](https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html) |
| 1D Kuramoto–Sivashinsky | `KS-v0` | C (partial) | D/C | 8 | 4 | Y | [Paper](https://royalsocietypublishing.org/doi/10.1098/rspa.2019.0351) |
| Lorenz63 | `Lorenz63-v0` | C | D/C | 3 | 3 | Y | [Paper](https://npg.copernicus.org/articles/31/319/2024/npg-31-319-2024.pdf); [Info](https://en.wikipedia.org/wiki/Lorenz_system) |
| Quadruple Gyre Flow | `QuadrupleGyreFlow-v0` | C | D/C | 3 | 2 | Y |  |
| Rössler | `Rossler-v0` | C | D/C | 3 | 3 | Y | [Info](http://www.scholarpedia.org/article/Rossler_attractor) |
| Hénon Map | `HenonMap-v0` | D/C | D/C | 2 | 2 | Y | [Info](https://en.wikipedia.org/wiki/H%C3%A9non_map) |
| Ikeda Map | `IkedaMap-v0` | C | D/C | 2 | 1 | Y | [Info](https://en.wikipedia.org/wiki/Ikeda_map) |
| Logistic Map | `LogisticMap-v0` | D/C | D/C | 1 | 1 | Y | [Paper](https://pubs.aip.org/aip/cha/article/9/3/775/136623/Optimal-chaos-control-through-reinforcement); [Info](https://blbadger.github.io/logistic-map.html#mjx-eqn-eq1) |
| Tent Map | `TentMap-v0` | D/C | D/C | 1 | 1 | Y | [Info](https://en.wikipedia.org/wiki/Tent_map) |
| Tinkerbell Map | `TinkerbellMap-v0` | C | D/C | 2 | 2 | N | [Info](https://www.starsandus.com/post/the-tinkerbell-map) |
| Fluidic Pinball | `FluidicPinball-v0` | C (partial) | D/C | 360 | 3 | Y |  |
| Kolmogorov Flow (HydroGym) | `HydroGym-Kolmogorov-v0` | C (partial) | D/C | 64 | 4 | Y | [Code](https://github.com/dynamicslab/hydrogym); [Paper](https://proceedings.mlr.press/v283/lagemann25a.html) |
| Turbulent Channel Flow (HydroGym) | `HydroGym-Channel-v0` | C (partial) | C | 128 | 24 | Y | [Code](https://github.com/dynamicslab/hydrogym); [Paper](https://proceedings.mlr.press/v283/lagemann25a.html) |
| Hopper (MJX) | `Hopper-v0` | C (partial) | D/C | 11 | 3 | Y | [Code](https://gymnasium.farama.org/environments/mujoco/hopper/); [Info](https://mujoco.readthedocs.io/en/stable/mjx.html) |
| Walker2d (MJX) | `Walker2d-v0` | C (partial) | D/C | 17 | 6 | Y | [Code](https://gymnasium.farama.org/environments/mujoco/walker2d/); [Info](https://mujoco.readthedocs.io/en/stable/mjx.html) |
| AYS Integrated Assessment Model | `AYSIAM-v0` | C | D/C | 3 | 2 | Y | [Code](https://github.com/fstrnad/pyDRLinWESM) |

For CFD-scale flow control we wrap [HydroGym](https://github.com/dynamicslab/hydrogym) rather than reimplementing it.
The `HydroGym-*` envs expose HydroGym's pure JAX environments through the BifurcaGym API (jit, vmap and all wrappers)
and need `pdm install -G hydrogym`. HydroGym's Firedrake, MAIA and NEK5000 environments cannot be jitted, so use
HydroGym directly for those. `HydroGym-Kolmogorov-v0` adds a seeded initial vorticity perturbation (`init_noise`, 0
reproduces HydroGym's fixed reset), and `HydroGym-Channel-v0` downloads its initial field from HuggingFace on first use.

`Hopper-v0` and `Walker2d-v0` port Gymnasium's MuJoCo v5 tasks to [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
and need `pdm install -G mujoco`. Rewards, termination, frame skip, reset noise and observations follow the Gymnasium
defaults, except that the torso pitch is wrapped to [-pi, pi] and the solver warm start is not carried between steps.
The discrete action versions are bang bang, with each joint torque in {0, +max, -max}. Expect around 40 s of jit
compilation per env on CPU.

Further environments (non-stationary boat in a current, multi-agent AYS, an ROV navigating a chaotic flow) are work in
progress and are not yet registered; see [ROADMAP.md](ROADMAP.md).

## Baselines

```bash
pdm run python baselines/ppo_baseline.py --env Pendulum-v0 --cont-action --total-timesteps 1000000
```

## Testing

```bash
pdm run test        # all registered envs, skips slow (FluidicPinball, HydroGym-Channel, Hopper, Walker2d) and render tests
pdm run test-all    # everything
```

These run on the CPU with 4 workers, limited threads per worker and a 16 GB memory cap, using `systemd-run` so Linux only. Elsewhere call `pytest -n 4 --dist load` directly, the thread limits are set in `tests/conftest.py`.

## Citation

If you use BifurcaGym in your work, please cite:

```bibtex
@misc{bifurcagym,
  title  = {BifurcaGym: A Benchmark of Reinforcement Learning and Control Environments with Chaotic Dynamics},
  author = {TODO},
  year   = {2026},
}
```

## License

MIT
