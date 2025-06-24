import jax
import jax.numpy as jnp
import jax.random as jrandom
import bifurcagym.registration
import matplotlib.pyplot as plt
import sys


env_names = [
             # "Acrobot-v0",
             # "CartPole-v0",
             # "NCartPole-v0",
             # "Pendubot-v0",
             "Pendulum-v0",
             # "WetChicken-v0",
             #  "KS-v0",
             # "LogisticMap-v0",
             # "TentMap-v0",
             ]

key = jrandom.key(42)

env = bifurcagym.make(env_names[0],
                      cont_state=True,
                      cont_action=True,
                      normalised=False,
                      autoreset=False,
                      metrics=False)

resolution = 200
actions = jnp.linspace(env.action_space().low, env.action_space().high, resolution)
obs = jnp.linspace(env.observation_space().low, env.observation_space().high, resolution)
# TODO should this perhaps be some kind of combination, for example maybe max state[0] and min state[1] my be largely different
# TODO the above kind of assumes some smoothness in the reward landscape
state = jax.vmap(env.get_state)(obs)

reward = jax.vmap(jax.vmap(env.reward_function, in_axes=(0, None, None, None)), in_axes=(None, None, 0, None))(actions, jnp.zeros(()), state, key)
print(reward.min(), reward.max())

sys.exit()
""" Below is specifically for cartpole """
def reward_func(theta, x):
    goal = jnp.array([0.0, env.length])
    pole_x = env.length * jnp.sin(theta)
    pole_y = env.length * jnp.cos(theta)
    position = jnp.array([x + pole_x, pole_y])
    squared_distance = jnp.sum((position - goal) ** 2)
    squared_sigma = 0.25 ** 2
    costs = 1 - jnp.exp(-0.5 * squared_distance / squared_sigma)
    return -costs

thetas = jnp.linspace(env.observation_space().low[0], env.observation_space().high[0], resolution)
xs = jnp.linspace(env.observation_space().low[2], env.observation_space().high[2], resolution)
reward = jax.vmap(jax.vmap(reward_func, in_axes=(0, None)), in_axes=(None, 0))(thetas, xs)
plt.figure(figsize=(8, 6))
c = plt.pcolormesh(xs, thetas, reward, cmap='viridis', shading='auto')
plt.colorbar(c, label='Reward')
plt.xlabel('X')
plt.ylabel('Theta')
plt.show()
