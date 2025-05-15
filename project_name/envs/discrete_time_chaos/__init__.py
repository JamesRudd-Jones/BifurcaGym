from project_name.envs.discrete_time_chaos import henon_map, logistic_map, tent_map


# HenonMap = henon_map.HenonMap
LogisticMapDSDA = logistic_map.LogisticMapDSDA
LogisticMapCSDA = logistic_map.LogisticMapCSDA
LogisticMapCSCA = logistic_map.LogisticMapCSCA
# TentMap = tent_map.TentMap


__all__ = ["LogisticMapDSDA",
           "LogisticMapCSDA",
           "LogisticMapCSCA"
           ]