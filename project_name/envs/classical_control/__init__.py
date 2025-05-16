from project_name.envs.classical_control import pendulum, pilco_cartpole, wet_chicken


PendulumCSDA = pendulum.PendulumCSDA
PendulumCSCA = pendulum.PendulumCSCA
PilcoCartPoleCSDA = pilco_cartpole.PilcoCartPoleCSDA
PilcoCartPoleCSCA = pilco_cartpole.PilcoCartPoleCSCA
# WetChicken = wet_chicken.WetChicken


__all__ = ["PendulumCSDA",
           "PendulumCSCA",
           "PilcoCartPoleCSDA",
           "PilcoCartPoleCSCA",
           # "WetChicken",
           ]