from bifurcagym.envs.classical_control import (acrobot,
                                               pendulum,
                                               pilco_cartpole,
                                               wet_chicken)

from bifurcagym.envs.continuous_time_chaos import (kuramoto_sivashinsky)

from bifurcagym.envs.discrete_time_chaos import (logistic_map,
                                                 henon_map,
                                                 tent_map,
                                                 )
from bifurcagym.wrappers import (AutoResetWrapper,
                                 NormalisedWrapperCSDA,
                                 NormalisedWrapperCSCA,
                                 PeriodicWrapper,
                                 )


def make(env_id: str,
         cont_state=False,
         cont_action=False,
         normalised=False,
         autoreset=False,
         **env_kwargs):

    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered environments.")

    # # # Classical Control
    ####################################################################################################################
    if env_id == "Acrobot-v0":
        if cont_state and cont_action:
            env = acrobot.AcrobotCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = acrobot.AcrobotCSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State versions.")

    elif env_id == "Pendulum-v0":
        if cont_state and cont_action:
            env = pendulum.PendulumCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = pendulum.PendulumCSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State versions.")

    elif env_id == "PilcoCartPole-v0":
        if cont_state and cont_action:
            env = pilco_cartpole.PilcoCartPoleCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = pilco_cartpole.PilcoCartPoleCSCA(**env_kwargs)
        else:
            raise ValueError("No Discrete State versions.")
    # elif env_id == "WetChicken-v0":
    #     env = wet_chicken.WetChicken(**env_kwargs)

    # # # Continuous Time Chaos
    ####################################################################################################################
    elif env_id == "KS-v0":
        if cont_state and cont_action:
            env = kuramoto_sivashinsky.KuramotoSivashinskyCSCA(**env_kwargs)
        else:
            raise ValueError("No Discrete State or Discrete Action versions.")


    # # # Discrete Time Chaos
    ####################################################################################################################
    elif env_id == "LogisticMap-v0":
        if cont_state and cont_action:
            env = logistic_map.LogisticMapCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = logistic_map.LogisticMapCSDA(**env_kwargs)
        elif not cont_state and not cont_action:
            env = logistic_map.LogisticMapDSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State Continuous Action version.")

    elif env_id == "HenonMap-v0":
        if cont_state and cont_action:
            env = henon_map.HenonMapCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = henon_map.HenonMapCSDA(**env_kwargs)
        elif not cont_state and not cont_action:
            env = henon_map.HenonMapDSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State Continuous Action version.")

    elif env_id == "TentMap-v0":
        if cont_state and cont_action:
            env = tent_map.TentMapCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = tent_map.TentMapCSDA(**env_kwargs)
        elif not cont_state and not cont_action:
            env = tent_map.TentMapDSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State Continuous Action version.")

    else:
        raise ValueError("Environment ID is not registered.")

    # # # Periodic wrapper that auto checks if env has periodic dimensions
    ####################################################################################################################
    if hasattr(env, 'periodic_dim') and env.periodic_dim is not None:
        env = PeriodicWrapper(env)

    # # # Normalises the observation and action space  # TODO can we do reward space as well?
    ####################################################################################################################
    if normalised:
        if not cont_state and not cont_action:
            raise ValueError("Can't Normalise Discrete State Discrete Action.")
        elif cont_state and not cont_action:
            env = NormalisedWrapperCSDA(env)
        elif cont_state and cont_action:
            env = NormalisedWrapperCSCA(env)
        else:
            raise ValueError("No Normalise Wrapper for Discrete State Continuous Action.")

    # # # Enables an autoresetting environment that works well with Jax, but not necessary for a for loop with conditionals that can't be Jitted
    ####################################################################################################################
    if autoreset:
        env = AutoResetWrapper(env)

    return env


registered_envs = ["Acrobot-v0",
                   "Pendulum-v0",
                   "PilcoCartPole-v0",
                   "WetChicken-v0",
                   "KS-v0",
                   "HenonMap-v0",
                   "LogisticMap-v0",
                   "TentMap-v0",
                   ]