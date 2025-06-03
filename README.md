# BifurcaGym

## Maybe a better name?

Motivation for these kinds of env, what RL approahes this is good for, and why in Jax

Environments can come in various discrete/continuous state and action space combinations, which can be easily selected 
when loading the environment. Wrappers are made for normalising state and action spaces as well as auto-resetting the 
environment to enable easy Jax based environment rollouts using scan. 

This is a wrapper since many of these environments are non-episodic and so we can save some computation from not having
to reset the env at every step. Further if you are using non Jittable training loops then using specific conditionals 
that break with a Done flag is also easily done.

Outputs are delta obs for Model-Based setups. This also has a wrapper and a function called get_delta_obs that ensures 
correct periodicity is applied for certain environments if you are using a dynamics model to predict the transition 
change rather than just the next observation.

We also easily allow a generative step so that obs can be fed into the env, again useful for some Model-Based RL setups.
Obs that are fed in must match the scale (aka if normalised or not) compared to the environment output.

Further, each env must have a defined reward function that again is amenable to Model-Based RL experiments where focus 
is placed on learning a representative transition function.

The benefit is that rewards can easily be adjusted by the user. We have tried to keep it comprehensive in that reward 
functions require the action taken (good for many control tasks where we want to minimise a control signal), the 
original state, the transitioned to state (allows easy one step state comparison), plus the definition of a key if 
stochatic rewards are beneficial. However, thought needs to go into how the random process is managed if comparing 
rewards from the environment with rewards calculated from collected trajectories. A benefit of Jax though is it can be 
fairly easy to store and track random keys to that this would be possible if needed.

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

- I think there way may be a better way to do the wrappers rather than so much duplicate code
- Should we have state space alongside the observation space?
- Add rendering for all the envs 
- Would be good to work with MARL and SARL - figure out how to easily fit in both with wrappers as well
- More verification of environments to reality, unsure exactly how to do this

Add the following envs:
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