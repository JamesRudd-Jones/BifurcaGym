from project_name.envs import (base_env,
                               classical_control,
                               continuous_time_chaos,
                               discrete_time_chaos,
                               fluid_control,
                               multi_agent_control)
from project_name.envs.discrete_time_chaos import logistic_map

# Pendulum = classical_control.Pendulum
# PilcoCartpole = classical_control.PilcoCartpole
# WetChicken = classical_control.WetChicken
LogisticMapDSDA = logistic_map.LogisticMapDSDA
LogisticMapCSDA = logistic_map.LogisticMapCSDA
LogisticMapCSCA = logistic_map.LogisticMapCSCA


__all__ = ["LogisticMapDSDA",
           "LogisticMapCSDA",
           "LogisticMapCSCA",
            #"Pendulum",
           # "PilcoCartpole",
           # "WetChicken",
           ]