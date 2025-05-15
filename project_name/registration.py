# from project_name.envs.classical_control import (pendulum,
#                                                  pilco_cartpole,
#                                                  wet_chicken)

from project_name.envs.discrete_time_chaos import (logistic_map,
                                                   )


def make(env_id: str, cont_state, cont_action, **env_kwargs):
    # TODO add some thanks to gymnax
    """A JAX-version of OpenAI's infamous env.make(env_name).


    Args:
      env_id: A string identifier for the environment.
      **env_kwargs: Keyword arguments to pass to the environment.


    Returns:
      A tuple of the environment and the default parameters.
    """
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered xxx environments.")
    # TODO fill the above with package name

    # # Classic Control
    # if env_id == "Pendulum-v0":
    #     env = pendulum.Pendulum(**env_kwargs)
    # elif env_id == "PilcoCartPole-v0":
    #     env = pilco_cartpole.PilcoCartPole(**env_kwargs)
    # elif env_id == "WetChicken-v0":
    #     env = wet_chicken.WetChicken(**env_kwargs)

    # Discrete Time Chaos
    elif env_id == "LogisticMap-v0":
        if cont_state and cont_action:
            env = logistic_map.LogisticMapCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = logistic_map.LogisticMapCSDA(**env_kwargs)
        elif not cont_state and not cont_action:
            env = logistic_map.LogisticMapDSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State Continuous Action version.")
    else:
        raise ValueError("Environment ID is not registered.")

    # Create a jax PRNG key for random seed control
    return env


registered_envs = ["Pendulum-v0",
                   "PilcoCartPole-v0",
                   "WetChicken-v0",
                   "LogisticMap-v0",
                   ]