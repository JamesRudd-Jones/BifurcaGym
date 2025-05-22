# BifurcaGym

## Maybe a better name?

How do we define discrete or continuous actions, some discretisation scheme easily?
Good to explain that here

Would be good to work with MARL and SARL - figure out how to easily fit in both with wrappers as well

What is the reward schema, is it controlling chaos or else, maybe easily adjustable by having a sep reward func, also good for model-based rl

BifurcaGym has the following environments:

| Environment    |     State Space     |    Action Space     |                                                                                            Reference                                                                                             |
|:---------------|:-------------------:|:-------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Logistic Map   | Discrete/Continuous | Discrete/Continuous |            [Info](https://blbadger.github.io/logistic-map.html#mjx-eqn-eq1); [Paper](https://pubs.aip.org/aip/cha/article/9/3/775/136623/Optimal-chaos-control-through-reinforcement)            |
| Tent Map       | Discrete/Continuous | Discrete/Continuous |                                                                          [Info](https://en.wikipedia.org/wiki/Tent_map)                                                                          |
| Pendulum       |     Continuous      | Discrete/Continuous |                                                                  [Code](https://github.com/fusion-ml/trajectory-information-rl)                                                                  |
| Pilco Cartpole |     Continuous      | Discrete/Continuous |                   [Code](https://github.com/fusion-ml/trajectory-information-rl); [Paper](https://aiweb.cs.washington.edu/research/projects/aiweb/media/papers/tmpZj4RyS.pdf)                    |
| Wet Chicken    | Continuous| Discrete/Continuous| [Code](https://github.com/LAVA-LAB/improved_spi/blob/main/wetChicken.py); [Paper](https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Publikationen/conferences_int/2009/Hans-ICANN-2009.pdf) |

Would be great to have a render option for all that can be done in eval stages, or perhaps some jittable gif saver?


# TODOs

I think there way may be a better way to do the wrappers rather than so much duplicate code
Should we have state space alongside the observation space?

Add the following envs:
- Acrobot
- Wet Chicken
- Lorenz xx
- Mackey Glass
- Van Der Pol Oscillator
- Predator Prey / Lotka-Volterra
- Henon Map
- Kuramoto Oscillator
- Double Pendulum
- N Pendulum
- Three Body Problem
- AYS IAM SARL
- AYS IAM MARL
- Fusion Problems