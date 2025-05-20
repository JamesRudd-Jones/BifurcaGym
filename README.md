# Chaotax

## Ideally a better name, chaoticgym?, others?

Current envs that work:
- Pendulum
- Pilco Cartpole
- Logistic Map

How do we define discrete or continuous actions, some discretisation scheme easily?
Good to explain that here

Would be good to work with MARL and SARL - figure out how to easily fit in both with wrappers as well

What is the reward schema, is it controlling chaos or else, maybe easily adjustable by having a sep reward func, also good for model-based rl

What envs do we provide:
- Logistic Map
- Tent Map
- Lorenz etc etc
- Predator-Prey
- Van der Pol Oscillator
- KS Equation
- Kuramoto Oscillator
- Other fluidy things
- N-Pendulum (aka like double pendulum and greater)
- Three body problem

Check the PhD thesis to get a summary of all the envs

Would be great to have a render option for all that can be done in eval stages, or perhaps some jittable gif saver?

Having some wrappers be ideal, for example some that normalise the env etc etc


# TODOs

Update all reward functions to be separate and take in state and new_state so can be used with a model-based setup
Should we have state space and observation space as well?