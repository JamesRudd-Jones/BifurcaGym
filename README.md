# BifurcaGym

Chaotic transition dynamics are found in many natural processes such as fluids, weather, financial systems, ecology, and 
even multi-agent systems, amongst many others.

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

Every environment is differentiable: `step` does not block gradients, so `jax.grad`, `jax.jvp` and `jax.vjp` work
through the dynamics, for differentiable simulation, analytic policy gradients or gradient based model predictive
control. Expect gradients through chaotic dynamics to grow exponentially over long horizons, and clipping or
termination to give zero or undefined gradients at their edges.

### Precision

`make` runs in 64-bit by default. Pass `x64=False` for 32-bit, which is available for Pendulum, CartPole, WetChicken,
FluidicPinball, HydroGym-Kolmogorov, Hopper and Walker2d. The chaotic systems need float64, so `make` raises
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
| N Cartpole | `NCartPole-v0` | C | D/C | 6 | 1 | Y | [Info](https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html) |
| N Pendulum | `NPendulum-v0` | C | D/C | 4 | 1 | Y |  |
| Pendulum | `Pendulum-v0` | C | D/C | 2 | 1 | Y | [Code](https://github.com/fusion-ml/trajectory-information-rl) |
| Wet Chicken | `WetChicken-v0` | D/C | D/C | 2 | 2 | Y | [Code](https://github.com/LAVA-LAB/improved_spi/blob/main/wetChicken.py); [Paper](https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Publikationen/conferences_int/2009/Hans-ICANN-2009.pdf) |
| Arnold–Beltrami–Childress Flow | `ABCFlow-v0` | C | D/C | 3 | 3 | Y | [Info](https://en.wikipedia.org/wiki/Arnold%E2%80%93Beltrami%E2%80%93Childress_flow) |
| Bickley Jet Flow | `BickleyJetFlow-v0` | C | D/C | 3 | 2 | Y | [Info](https://en.wikipedia.org/wiki/Bickley_jet) |
| Chua | `Chua-v0` | C | D/C | 3 | 2 | Y | [Info](http://www.scholarpedia.org/article/Chua_circuit) |
| Double Gyre Flow (`n_gyre_rows` rows of 2 gyres) | `DoubleGyreFlow-v0` | C | D/C | 3 | 2 | Y | [Paper](https://www.nature.com/articles/s41467-021-27015-y.pdf); [Info](https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html) |
| 1D Kuramoto–Sivashinsky | `KS-v0` | C (partial) | D/C | 8 | 4 | Y | [Paper](https://royalsocietypublishing.org/doi/10.1098/rspa.2019.0351) |
| Lorenz63 | `Lorenz63-v0` | C | D/C | 3 | 3 | Y | [Paper](https://npg.copernicus.org/articles/31/319/2024/npg-31-319-2024.pdf); [Info](https://en.wikipedia.org/wiki/Lorenz_system) |
| R&ouml;ssler | `Rossler-v0` | C | D/C | 3 | 3 | Y | [Info](http://www.scholarpedia.org/article/Rossler_attractor) |
| Arnold's Cat Map | `CatMap-v0` | C | D/C | 2 | 2 | Y | [Info](https://en.wikipedia.org/wiki/Arnold%27s_cat_map) |
| H&eacute;non Map | `HenonMap-v0` | D/C | D/C | 2 | 2 | Y | [Info](https://en.wikipedia.org/wiki/H%C3%A9non_map) |
| Ikeda Map | `IkedaMap-v0` | C | D/C | 2 | 1 | Y | [Info](https://en.wikipedia.org/wiki/Ikeda_map) |
| Logistic Map | `LogisticMap-v0` | D/C | D/C | 1 | 1 | Y | [Paper](https://pubs.aip.org/aip/cha/article/9/3/775/136623/Optimal-chaos-control-through-reinforcement); [Info](https://blbadger.github.io/logistic-map.html#mjx-eqn-eq1) |
| Lozi Map | `LoziMap-v0` | C | D/C | 2 | 2 | Y | [Info](https://en.wikipedia.org/wiki/Lozi_map) |
| Skew Product (driver observed, or hidden with `observe_driver=False`) | `SkewProduct-v0` | C (partial if hidden) | D/C | 2 | 1 | Y |  |
| Standard Map | `StandardMap-v0` | C | D/C | 2 | 1 | Y | [Info](http://www.scholarpedia.org/article/Chirikov_standard_map) |
| Tent Map | `TentMap-v0` | D/C | D/C | 1 | 1 | Y | [Info](https://en.wikipedia.org/wiki/Tent_map) |
| Fluidic Pinball | `FluidicPinball-v0` | C (partial) | D/C | 360 | 3 | Y |  |
| Kolmogorov Flow (HydroGym) | `HydroGym-Kolmogorov-v0` | C (partial) | D/C | 64 | 4 | Y | [Code](https://github.com/dynamicslab/hydrogym); [Paper](https://proceedings.mlr.press/v283/lagemann25a.html) |
| Turbulent Channel Flow (HydroGym) | `HydroGym-Channel-v0` | C (partial) | C | 128 | 24 | Y | [Code](https://github.com/dynamicslab/hydrogym); [Paper](https://proceedings.mlr.press/v283/lagemann25a.html) |
| Hopper (MJX) | `Hopper-v0` | C (partial) | D/C | 11 | 3 | Y | [Code](https://gymnasium.farama.org/environments/mujoco/hopper/); [Info](https://mujoco.readthedocs.io/en/stable/mjx.html) |
| Walker2d (MJX) | `Walker2d-v0` | C (partial) | D/C | 17 | 6 | Y | [Code](https://gymnasium.farama.org/environments/mujoco/walker2d/); [Info](https://mujoco.readthedocs.io/en/stable/mjx.html) |

### Dynamics

Every environment is labelled on two axes: the role chaos plays in the task (suppress it, navigate through it,
swing up through it, chaos induced by control, or a non-chaotic reference), and the kind of
dynamical system (map, ODE or PDE; dissipative, conservative or non-invertible; autonomous or forced; hyperbolic or
not). The measured columns come from `bifurcagym.analysis` under zero control from a reset state: the verdict of
`classify_dynamics`, the leading Lyapunov exponent, the Lyapunov time (physical time, or steps for maps), how many
Lyapunov times fit in an episode, the Kaplan–Yorke dimension, and the returns of zero control and of random actions.

<!-- dynamics-table:start -->
| Env | Role | Structure | Energy | Forcing | Hyperbolicity | Measured | λ₁ / step | λ₁ / time | Lyapunov time | e-folds / episode | KY dim | Zero ctrl return | Random return |
|:--|:--|:--|:--|:--|:--|:--:|--:|--:|--:|--:|--:|--:|--:|
| `Acrobot-v0` | swing-up | ODE | conservative | autonomous | non-hyperbolic | regular | 9.96e-05 | 0.000498 | 2.01e+03 | 0.199 | 1.74 | -2e+03 | -2e+03 |
| `CartPole-v0` | reference | ODE | dissipative | autonomous | n/a | fixed point | 0 | 0 | – | 0 | 1 | 16.6 | 3.23 |
| `NCartPole-v0` | swing-up | ODE | dissipative | autonomous | non-hyperbolic | fixed point | 0 | 0 | – | 0 | 1 | 0.00463 | 0.0547 |
| `NPendulum-v0` | swing-up | ODE | conservative | autonomous | non-hyperbolic | mixed (1/4 chaotic) | 0.00636 | 0.127 | 7.86 | 6.36 | 3.99 | -1.57e+04 | -1.58e+04 |
| `Pendulum-v0` | reference | ODE | conservative | autonomous | n/a | regular | 0.000342 | 0.00683 | 146 | 0.342 | 2 | -6.04e+03 | -7.1e+03 |
| `WetChicken-v0` | reference | map | stochastic | autonomous | n/a | stochastic (not assessed) | – | – | – | – | – | -1.85e+03 | -1.77e+03 |
| `ABCFlow-v0` | navigate | ODE | conservative | autonomous | non-hyperbolic | mixed (2/4 chaotic) | 0.00215 | 0.043 | 23.2 | 21.5 | 3 | -100 | -91.9 |
| `BickleyJetFlow-v0` | navigate | ODE | conservative | forced | non-hyperbolic | mixed (2/4 chaotic) | 0.000861 | 0.00861 | 116 | 0.431 | 2 | 5 | 5 |
| `Chua-v0` | suppress | ODE | dissipative | autonomous | non-hyperbolic | chaotic | 0.00461 | 0.461 | 2.17 | 92.2 | 2.11 | 7.9e+03 | 7.88e+03 |
| `DoubleGyreFlow-v0` | navigate | ODE | conservative | forced | non-hyperbolic | chaotic | 0.0105 | 0.105 | 9.5 | 21.1 | 2 | 4.93 | -3.33 |
| `KS-v0` | suppress | PDE | dissipative | autonomous | non-hyperbolic | chaotic | 0.00278 | 0.0556 | 18 | 27.8 | – | -1.41e+05 | -1.2e+05 |
| `Lorenz63-v0` | suppress | ODE | dissipative | autonomous | singular hyperbolic | chaotic | 0.00896 | 0.896 | 1.12 | 179 | 2.06 | 7.41e+03 | 7.36e+03 |
| `Rossler-v0` | suppress | ODE | dissipative | autonomous | non-hyperbolic | chaotic | 0.00113 | 0.0565 | 17.7 | 11.3 | 2.01 | 4.56e+03 | 4.4e+03 |
| `HenonMap-v0` | suppress | map | dissipative | autonomous | non-hyperbolic | chaotic | 0.42 | – | 2.38 | 420 | 1.26 | 466 | 460 |
| `IkedaMap-v0` | suppress | map | dissipative | autonomous | non-hyperbolic | chaotic | 0.514 | – | 1.95 | 1.03e+03 | 1.71 | 871 | 177 |
| `LogisticMap-v0` | suppress | map | non-invertible | autonomous | non-hyperbolic | chaotic | 0.431 | – | 2.32 | 129 | 1 | -21.1 | -23.1 |
| `LoziMap-v0` | suppress | map | dissipative | autonomous | hyperbolic | chaotic | 0.468 | – | 2.14 | 468 | 1.4 | 433 | 434 |
| `TentMap-v0` | suppress | map | non-invertible | autonomous | hyperbolic | chaotic | 0.693 | – | 1.44 | 346 | 1 | -54.8 | -49.3 |
| `CatMap-v0` | suppress | map | conservative | autonomous | hyperbolic | chaotic | 0.962 | – | 1.04 | 962 | 2 | 393 | 393 |
| `StandardMap-v0` | suppress | map | conservative | autonomous | non-hyperbolic | chaotic | 0.133 | – | 7.53 | 133 | 2 | 438 | 317 |
| `SkewProduct-v0` | navigate | map | non-invertible | forced | hyperbolic | chaotic | 0.688 | – | 1.45 | 688 | 1.99 | 376 | 362 |
<!-- dynamics-table:end -->

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

Further environments (the AYS integrated assessment model, single and multi-agent, a non-stationary boat in a
current, and an ROV navigating a chaotic flow) are work in progress and are not yet registered. Pendubot (the same
problem as `NPendulum-v0` with two links) and the Tinkerbell map (a third map in the same role as Hénon and Ikeda) were
retired from the suite to avoid duplication. They live in `experimental/`, outside the installed package, and use the
optional `experimental` dependency group; see [ROADMAP.md](ROADMAP.md).

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
  author = {James Rudd-Jones and Mirco Musolesi and Mar{\'\i}a P{\'e}rez-Ortiz},
  year   = {2026},
}
```

## License

MIT
