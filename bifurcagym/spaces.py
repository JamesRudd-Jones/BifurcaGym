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
        self.n = num_discrete
        self.shape = (self.n,)
        # self.shape = ()
        # Trialling this to make it easier from the RL side of things rather than having to do
        # self.observation_space().shape[0] for continuous and self.observation_space().n for discrete
        # If it causes issues then can easily change it back
        self.dtype = jnp.int_

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jrandom.randint(key, shape=(1,), minval=0, maxval=self.n).astype(self.dtype)


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
