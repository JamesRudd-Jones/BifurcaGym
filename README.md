# BifurcaGym

Chaotic transition dynamics are found in many natural processes such as fluids, weather, financial systems, ecology and even multi-agent systems.

In many of these scenarios we want to learn behavioural policies towards a desired objective, such as in fluid control (e.g. fluid mixing, aerofoil optimisation), financial or ecosystem control (a stable desired equilibrium), weather model parameter optimisation, autonomous vehicle control under wind/current effects (e.g. underwater drones or sailboats).

However, chaos has a huge impact on Reinforcement Learning or Optimal Control methods as we look to optimise for long horizon sequential tasks.

Due to chaotic dynamics having sensitivity to initial conditions, it can be hard to ensure robustness and accuracy in prediction and control in these types of environments.

Bifurcagym is a collection of environments that experience chaos in their transition dynamics, so that practitioners and theorists can understand the impact chaos has on their algorithm design. 

Bifurcagym is written in Jax to gain the benefits of the enhanced Autodiff and acceleration potential not only for the non-linear environment dynamics (which many require extensive CFD/FEA/FEM solvers) but also in the reduction of overheads between the environment and the agent for Reinforcement Learning/Optimal Control. 

## The Framework

Since both Model-Based and Model-Free methods are typically used for these types of environments, we ensure that Bifurcagym environments are set up to provide the correct outputs and wrappers for both. For example outputs also contain delta obs (the change in observation between steps) as this is amenable for Model-Based setups. This also has a wrapper and a function called get_delta_obs that ensures correct periodicity is applied for certain environments if you are using a dynamics model to predict the transition change rather than just the next observation.

Environments can come in various discrete/continuous state and action space combinations, which can be easily selected when loading the environment. Wrappers are made for normalising state and action spaces as well as auto-resetting the environment to enable easy Jax based environment rollouts using jax.lax.scan. 

We also easily allow a generative step so that obs can be fed into the env, again useful for some Model-Based RL setups that utilise generative dynamics models.
Obs that are fed in must match the scale (aka if normalised or not) compared to the environment output.

Further, each env must have a defined reward function that again is amenable to Model-Based RL experiments where focus is placed on learning a transition function.

BifurcaGym has the following environments currently:

| Environment             |     State Space     |    Action Space     |                                                                                            Reference                                                                                             |
|:------------------------|:-------------------:|:-------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Acrobot                 |     Continuous      | Discrete/Continuous |         [Code](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/acrobot.py); [Paper](http://users.cms.caltech.edu/~murray/preprints/erl-M91-46.pdf)          |
| Cartpole                |     Continuous      | Discrete/Continuous |                                      [Code](https://github.com/fusion-ml/trajectory-information-rl); [Paper](https://ieeexplore.ieee.org/document/6313077)                                       |
| N Cartpole              |     Continuous      | Discrete/Continuous |                                                       [Info](https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html)                                                       |
| Pendubot                |     Continuous      | Discrete/Continuous |                                        [Paper](https://link.springer.com/chapter/10.1007/BFb0015081); [Info](https://underactuated.mit.edu/acrobot.html)                                         |
| Pendulum                |     Continuous      | Discrete/Continuous |                                                                  [Code](https://github.com/fusion-ml/trajectory-information-rl)                                                                  |
| Wet Chicken             | Discrete/Continuous | Discrete/Continuous| [Code](https://github.com/LAVA-LAB/improved_spi/blob/main/wetChicken.py); [Paper](https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Publikationen/conferences_int/2009/Hans-ICANN-2009.pdf) |
| 1D Kuramoto-Sivashinsky | Continuous | Continuous|                                                              [Paper](https://royalsocietypublishing.org/doi/10.1098/rspa.2019.0351)                                                              |
| Logistic Map            | Discrete/Continuous | Discrete/Continuous |            [Info](https://blbadger.github.io/logistic-map.html#mjx-eqn-eq1); [Paper](https://pubs.aip.org/aip/cha/article/9/3/775/136623/Optimal-chaos-control-through-reinforcement)            |
| Tent Map                | Discrete/Continuous | Discrete/Continuous |                                                                          [Info](https://en.wikipedia.org/wiki/Tent_map)                                                                          |







# TODOs

- Check why normalisation seems to affect results negatively
- Do we still want the reward space so that we can normalise rewards, can we really validate this?
- Improve metrics wrapper so it doesn't add .env_state.env_state as this is confusing
- Would be good to work with MARL and SARL - figure out how to easily fit in both with wrappers as well
- Create utils for certain env types, e.g. discrete time chaos may all share a projection func
- The benefit is that rewards can easily be adjusted by the user. We have tried to keep it comprehensive in that reward 
functions require the action taken (good for many control tasks where we want to minimise a control signal), the 
original state, the transitioned to state (allows easy one step state comparison), plus the definition of a key if 
stochatic rewards are beneficial. However, thought needs to go into how the random process is managed if comparing 
rewards from the environment with rewards calculated from collected trajectories. A benefit of Jax though is it can be 
fairly easy to store and track random keys to that this would be possible if needed.
- Should we add back in params outside the env? Cons of this approach are ineffective calculation of params, as in
it is unclear if can do params 1 * param 2 and then store this so don't have to repeat the calc. Pros are perhaps for
non-stationary envs then it may be way easier to define params outside the env so they can change in a functional loop.
- Do we want a vmap wrapper? Used to have one but removed as felt fairly pointless

Add the following envs:
- N Link Pendulum
- Bernoulli Map/Dyadic Map
- LorenzXX
- Mackey Glass
- Van Der Pol Oscillator
- Predator Prey / Lotka-Volterra
- Henon Map
- Kuramoto Oscillator
- Three Body Problem
- AYS IAM SARL
- AYS IAM MARL
- Fusion Problems
- Wake flow control
- 3 pinballs or something similar to emulate wind turbine fields
- Something more complex than KS equations
