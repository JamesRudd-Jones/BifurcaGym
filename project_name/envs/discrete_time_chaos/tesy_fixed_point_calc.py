import jax
import jax.numpy as jnp
import jax.scipy as jsp


r = 3.1


def fixed_point_finder(r, period=1):
    def logistic_func(x):
        return r * x * (1 - x)

    if period == 1:
        period_1_root = (r - 1) / r
    else:
        raise ValueError('period must be 1 for now')

    return period_1_root

print(fixed_point_finder(r))