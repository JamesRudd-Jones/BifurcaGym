# Roadmap

- Check why normalisation seems to affect results negatively
- Improve metrics wrapper so it doesn't add .env_state.env_state as this is confusing
- Would be good to work with MARL and SARL - figure out how to easily fit in both with wrappers as well
- The benefit is that rewards can easily be adjusted by the user. We have tried to keep it comprehensive in that reward 
functions require the action taken (good for many control tasks where we want to minimise a control signal), the 
original state, the transitioned to state (allows easy one step state comparison), plus the definition of a key if 
stochastic rewards are beneficial. However, thought needs to go into how the random process is managed if comparing 
rewards from the environment with rewards calculated from collected trajectories. A benefit of Jax though is it can be 
fairly easy to store and track random keys to that this would be possible if needed.

Add the following envs:
- Mackey Glass
- Van Der Pol Oscillator
- Predator Prey / Lotka-Volterra
- Kuramoto Oscillator
- Three Body Problem
- AYS IAM SARL
- AYS IAM MARL
- Fusion Problems
- Wake flow control
- 3 pinballs or something similar to emulate wind turbine fields
- Something more complex than KS equations
