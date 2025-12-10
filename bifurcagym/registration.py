from bifurcagym.envs.classical_control import (acrobot,
                                               cartpole,
                                               n_cartpole,
                                               pendulum,
                                               wet_chicken)

from bifurcagym.envs.continuous_time_chaos import (bickley_jet_flow,
                                                   double_gyre_flow,
                                                   kuramoto_sivashinsky,
                                                   quadruple_gyre_flow,)

from bifurcagym.envs.discrete_time_chaos import (logistic_map,
                                                 henon_map,
                                                 tent_map,
                                                 )
from bifurcagym.envs.fluid_control import (fluidic_pinball,
                                           )
from bifurcagym.envs.non_stationary import (boat_in_current,
                                            )
from bifurcagym.wrappers import (AutoResetWrapper,
                                 MetricsWrapper,
                                 NormalisedWrapperCSDA,
                                 NormalisedWrapperCSCA,
                                 PeriodicWrapper,
                                 )


def make(env_id: str,
         cont_state=False,
         cont_action=False,
         normalised=False,
         autoreset=False,
         metrics=False,
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

    elif env_id == "CartPole-v0":
        if cont_state and cont_action:
            env = cartpole.CartPoleCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = cartpole.CartPoleCSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State versions.")

    elif env_id == "NCartPole-v0":
        if cont_state and cont_action:
            env = n_cartpole.NCartPoleCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = n_cartpole.NCartPoleCSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State versions.")

    elif env_id == "Pendubot-v0":
        if cont_state and cont_action:
            env = acrobot.PendubotCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = acrobot.PendubotCSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State versions.")

    # TODO add in n pendulum here

    elif env_id == "Pendulum-v0":
        if cont_state and cont_action:
            env = pendulum.PendulumCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = pendulum.PendulumCSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State versions.")

    elif env_id == "WetChicken-v0":
        if cont_state and cont_action:
            env = wet_chicken.WetChickenCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = wet_chicken.WetChickenCSDA(**env_kwargs)
        elif not cont_state and not cont_action:
            env = wet_chicken.WetChickenDSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State Continuous Action version.")

    # # # Continuous Time Chaos
    ####################################################################################################################
    elif env_id == "BickleyJetFlow-v0":
        if cont_state and cont_action:
            env = bickley_jet_flow.BickleyJetFlowCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = bickley_jet_flow.BickleyJetFlowCSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State or Discrete Action versions.")

    elif env_id == "DoubleGyreFlow-v0":
        if cont_state and cont_action:
            env = double_gyre_flow.DoubleGyreFlowCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = double_gyre_flow.DoubleGyreFlowCSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State or Discrete Action versions.")

    elif env_id == "KS-v0":
        if cont_state and cont_action:
            env = kuramoto_sivashinsky.KuramotoSivashinskyCSCA(**env_kwargs)
        else:
            raise ValueError("No Discrete State or Discrete Action versions.")

    elif env_id == "QuadrupleGyreFlow-v0":
        if cont_state and cont_action:
            env = quadruple_gyre_flow.QuadrupleGyreFlowCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = quadruple_gyre_flow.QuadrupleGyreFlowCSDA(**env_kwargs)
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

    # # # Fluid Control
    ####################################################################################################################
    elif env_id == "FluidicPinball-v0":
        if cont_state and cont_action:
            env = fluidic_pinball.FluidicPinballCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = fluidic_pinball.FluidicPinballCSDA(**env_kwargs)
        else:
            raise ValueError("No Discrete State or Discrete Action versions.")


    # # # Non-Stationary
    ####################################################################################################################
    elif env_id == "BoatInCurrent-v0":
        if cont_state and cont_action:
            env = boat_in_current.BoatInCurrentCSCA(**env_kwargs)
        elif cont_state and not cont_action:
            env = boat_in_current.BoatInCurrentCSCA(**env_kwargs)
        else:
            raise ValueError("No Discrete State version.")

    else:
        raise ValueError("Environment ID is not registered.")

    # # # Periodic wrapper that auto checks if env has periodic dimensions
    ####################################################################################################################
    if hasattr(env, 'periodic_dim') and env.periodic_dim is not None:
        env = PeriodicWrapper(env)

    # # # Normalises the observation, action, and reward space
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

    # # # Enables an autoresetting environment that works well with jax.lax.scan, but not necessary for a for loop with conditionals that can't be Jitted
    ####################################################################################################################
    if autoreset:
        env = AutoResetWrapper(env)

    if metrics:
        env = MetricsWrapper(env)

    return env


registered_envs = ["Acrobot-v0",
                   "CartPole-v0",
                   "NCartPole-v0",
                   "Pendubot-v0",
                   "Pendulum-v0",
                   "WetChicken-v0",
                   "BickleyJetFlow-v0",
                   "DoubleGyreFlow-v0",
                   "KS-v0",
                   "QuadrupleGyreFlow-v0",
                   "HenonMap-v0",
                   "LogisticMap-v0",
                   "TentMap-v0",
                   "FluidicPinball-v0",
                   "BoatInCurrent-v0"
                   ]