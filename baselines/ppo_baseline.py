import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax
import flax.linen as nn
import optax
from typing import Sequence, NamedTuple
import distrax
import bifurcagym
from flax.training.train_state import TrainState
import os
import sys
import wandb


print(f"Device: {jax.extend.backend.get_backend().platform}")


class ActorCriticDiscrete(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(64)(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(64)(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(64)(x)
        critic = nn.relu(critic)
        critic = nn.Dense(64)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, critic.squeeze(axis=-1)


class ActorCriticContinuous(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(64)(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(64)(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(64)(x)
        critic = nn.relu(critic)
        critic = nn.Dense(64)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, critic.squeeze(axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    env_state: bifurcagym.envs.base_env.EnvState


env_names = [
             # "Acrobot-v0",
             "CartPole-v0",
             # "NCartPole-v0",
             # "Pendubot-v0",
             # "Pendulum-v0",
             # "WetChicken-v0",
             #  "KS-v0",
             # "LogisticMap-v0",
             # "TentMap-v0",
             ]

key = jrandom.key(42)

config = {"LR": 1e-3,
          "UPDATE_EPOCHS": 4,
          "NUM_MINIBATCHES": 4,
          "GAMMA": 0.99,
          "GAE_LAMBDA": 0.95,
          "CLIP_EPS": 0.2,
          "ENT_COEF": 0.01,
          "VF_COEF": 0.5,
          "MAX_GRAD_NORM": 1.0,
          "ANNEAL_LR": True,
          "DEBUG": True,
          }

# total_timesteps = 100000000
total_timesteps = 70000000
num_envs = 64
rollout_length = 128
num_updates = total_timesteps // rollout_length // num_envs
config["MINIBATCH_SIZE"] = (num_envs * rollout_length // config["NUM_MINIBATCHES"])

cont_state = True
cont_action = False
normalised = False
autoreset = True
metrics = True

wandb.init(project="bifurcagym_testing",
           group=env_names[0],
           name=f"Cont State: {cont_state}; Cont Action: {cont_action}; Normalised: {normalised}; AutoReset: {autoreset}; Metrics: {metrics}",
           # mode="disabled",
           config=config,
           )

env = bifurcagym.make(env_names[0],
                      cont_state=cont_state,
                      cont_action=cont_action,
                      normalised=normalised,
                      autoreset=autoreset,
                      metrics=metrics)
key, _key = jrandom.split(key)
batch_key = jrandom.split(_key, num_envs)
init_obs, init_env_state = jax.vmap(env.reset)(batch_key)

if cont_action:
    network = ActorCriticContinuous(env.action_space().shape[0])
else:
    network = ActorCriticDiscrete(env.action_space().shape[0])
init_x = jnp.zeros(env.observation_space().shape)
key, _key = jrandom.split(key)
network_params = network.init(_key, init_x)

def linear_schedule(count):
    frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
    return config["LR"] * frac

if config["ANNEAL_LR"]:
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                     optax.adam(learning_rate=linear_schedule, eps=1e-5))
else:
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                     optax.adam(config["LR"], eps=1e-5))

train_state = TrainState.create(apply_fn=network.apply,
                                params=network_params,
                                tx=tx,
                                )

def update_step(runner_state, unused):
    def _env_step(runner_state, unused):
        train_state, env_state, obs, key = runner_state

        key, _key = jrandom.split(key)
        pi, value = train_state.apply_fn(train_state.params, obs)
        action = pi.sample(seed=_key)
        log_prob = pi.log_prob(action)

        # STEP ENV
        key, _key = jrandom.split(key)
        batch_key = jrandom.split(_key, num_envs)
        nobs, delta_obs, nenv_state, reward, done, info = jax.vmap(env.step)(action, env_state, batch_key)
        transition = Transition(done, action, value, reward, log_prob, obs, info, env_state)
        return (train_state, nenv_state, nobs, key), transition

    runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, rollout_length)
    train_state, env_state, last_obs, key = runner_state

    _, last_val = train_state.apply_fn(train_state.params, last_obs)

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            delta = transition.reward + config["GAMMA"] * next_value * (1 - transition.done) - transition.value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(_get_advantages,
                                     (jnp.zeros_like(last_val), last_val),
                                     traj_batch,
                                     reverse=True,
                                     unroll=16,
                                     )
        return advantages, advantages + traj_batch.value

    advantages, targets = _calculate_gae(traj_batch, last_val)

    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info

            def _loss_fn(params, traj_batch, gae, targets):
                # RERUN NETWORK
                pi, value = network.apply(params, traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = (jnp.clip(ratio,
                                        1.0 - config["CLIP_EPS"],
                                        1.0 + config["CLIP_EPS"],
                                        ) * gae)
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()

                total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                return total_loss, (value_loss, loss_actor, entropy)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, loss_info), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss_info

        train_state, traj_batch, advantages, targets, key = update_state
        key, _key = jrandom.split(key)
        # Batching and Shuffling
        batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
        assert (batch_size == rollout_length * num_envs), "batch size must be equal to number of steps * number of envs"
        permutation = jax.random.permutation(_key, batch_size)
        batch = (traj_batch, advantages, targets)
        batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        # Mini-batch Updates
        minibatches = jax.tree.map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                                   shuffled_batch
                                   )
        train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)

        return (train_state, traj_batch, advantages, targets, key), total_loss

    # Updating Training State and Metrics:
    update_state = (train_state, traj_batch, advantages, targets, key)
    update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
    train_state = update_state[0]
    metric = traj_batch.info
    key = update_state[-1]

    # Debugging mode
    if config.get("DEBUG"):
        def callback(info, policy_loss, critic_loss):
            return_values = info["returned_episode_returns"][info["returned_episode"]]
            timesteps = info["timestep"][info["returned_episode"]] * num_envs
            step_dict = {}
            for t in range(len(timesteps)):
                step_dict["global step"] = timesteps[t]
                step_dict["episodic return"] = return_values[t]

            step_dict["policy loss"] = policy_loss.squeeze()
            step_dict["critic loss"] = critic_loss.squeeze()

            wandb.log(step_dict)

        jax.debug.callback(callback, metric, jnp.mean(loss_info[1]), jnp.mean(loss_info[0]))

    return (train_state, env_state, last_obs, key), traj_batch

key, _key = jrandom.split(key)
runner_state = (train_state, init_env_state, init_obs, _key)
runner_state, traj_batch = jax.lax.scan(update_step, runner_state, None, num_updates)
trained_state = runner_state[0]
metric = jax.tree.map(lambda x: x[-1], traj_batch.info)

def test_step(runner_state, unused):
    train_state, env_state, obs, key = runner_state

    key, _key = jrandom.split(key)
    pi, value = train_state.apply_fn(train_state.params, obs)
    action = pi.sample(seed=_key)
    log_prob = pi.log_prob(action)

    # STEP ENV
    key, _key = jrandom.split(key)
    nobs, delta_obs, nenv_state, reward, done, info = env.step(action, env_state, _key)
    transition = Transition(done, action, value, reward, log_prob, obs, info, env_state)
    return (train_state, nenv_state, nobs, key), transition

key, _key = jrandom.split(key)
init_obs, init_env_state = env.reset(_key)
key, _key = jrandom.split(key)
runner_state = (trained_state, init_env_state, init_obs, _key)
runner_state, test_traj_batch = jax.lax.scan(test_step, runner_state, None, 1000)

return_values = test_traj_batch.info["returned_episode_returns"][test_traj_batch.info["returned_episode"]]
timesteps = test_traj_batch.info["timestep"][test_traj_batch.info["returned_episode"]] * num_envs
for t in range(len(timesteps)):
    print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
# print(f"global step={timesteps[-1]}, episodic return={return_values[-1]}")

# # first dim is num updates, second is num steps, third is num env; the last one can be any choice but won't work if num_envs=1
# if metrics:
#     traj_for_rendering = test_traj_batch.env_state.env_state
# else:
#     traj_for_rendering = test_traj_batch.env_state
#
# if cont_action:
#     # env.render_traj(traj_for_rendering, file_path=f"../../animations/baselines/Cont-Action")
#     env.render_traj(traj_for_rendering, file_path=f"./animations/baselines/Cont-Action")
# else:
#     # env.render_traj(traj_for_rendering, file_path=f"../../animations/baselines/Discrete-Action")
#     env.render_traj(traj_for_rendering, file_path=f"./animations/baselines/Discrete-Action")
