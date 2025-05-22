from typing import Any
import jax.numpy as jnp
import jax.random as jrandom
import chex

class Space:
    def sample(self, key: chex.Array) -> chex.Array:
        raise NotImplementedError


class Discrete(Space):
    def __init__(self,
                 num_discrete: int):
        self.num_discrete = num_discrete
        self.shape = ()
        self.dtype = jnp.int_

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jrandom.randint(key, shape=self.shape, minval=0, maxval=self.num_discrete).astype(self.dtype)


class Box(Space):
    def __init__(self,
                 low: chex.Array | float,
                 high: chex.Array | float,
                 shape: Any,
                 dtype: jnp.dtype = jnp.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jrandom.uniform(key, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype)
