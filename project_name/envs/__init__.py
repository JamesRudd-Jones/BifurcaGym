from project_name.envs import (base_env,
                               classical_control,
                               continuous_time_chaos,
                               discrete_time_chaos,
                               fluid_control,
                               multi_agent_control)
from project_name.envs.discrete_time_chaos import (henon_map,
                                                   logistic_map,
                                                   tent_map)

# PendulumCSDA = classical_control.PendulumCSDA
# PendulumCSCA = classical_control.PendulumCSCA
# PilcoCartPoleCSDA = classical_control.PilcoCartPoleCSDA
# PilcoCartPoleCSCA = classical_control.PilcoCartPoleCSCA
# # WetChicken = classical_control.WetChicken
#
# HenonMapDSDA = henon_map.HenonMapDSDA
# HenonMapCSDA = henon_map.HenonMapCSDA
# HenonMapCSCA = henon_map.HenonMapCSCA
# LogisticMapDSDA = logistic_map.LogisticMapDSDA
# LogisticMapCSDA = logistic_map.LogisticMapCSDA
# LogisticMapCSCA = logistic_map.LogisticMapCSCA
#
#
# __all__ = ["PendulumCSDA",
#            "PendulumCSCA",
#            "PilcoCartPoleCSDA",
#            "PilcoCartPoleCSCA",
#            "HenonMapDSDA",
#            "HenonMapCSDA",
#            "HenonMapCSCA",
#            "LogisticMapDSDA",
#            "LogisticMapCSDA",
#            "LogisticMapCSCA",
#            # "WetChicken",
#            ]