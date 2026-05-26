import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
import chex
from typing import Any, Dict, Tuple
from bifurcagym.envs import EnvState, EnvParams
from flax import struct
import dataclasses


@struct.dataclass
class MetricsEnvState:
    env_state: Any
    episode_lengths: int
    episode_returns: float
    returned_episode_lengths: int
    returned_episode_returns: float
    timestep: int


# Global cache to prevent re-registering identical dataclasses with JAX/Flax
_AUGMENTED_STATE_CACHE = {}


def create_augmented_state_class(base_state_instance):
    """
    Dynamically creates a flat Flax dataclass containing all fields from the
    original sub-environment state plus the metrics fields.
    """
    base_cls = type(base_state_instance)
    if base_cls in _AUGMENTED_STATE_CACHE:
        return _AUGMENTED_STATE_CACHE[base_cls]

    # Extract all existing fields from the specific sub-environment state
    base_fields = dataclasses.fields(base_cls)

    # Define our extra metrics fields with default values
    metrics_fields = [
        ("episode_lengths", int, dataclasses.field(default=0)),
        ("episode_returns", float, dataclasses.field(default=0.0)),
        ("returned_episode_lengths", int, dataclasses.field(default=0)),
        ("returned_episode_returns", float, dataclasses.field(default=0.0)),
        ("timestep", int, dataclasses.field(default=0)),
    ]

    # Combine them into a new flat dataclass definition
    # Note: Fields with defaults must come after fields without defaults.
    combined_fields = []
    for f in base_fields:
        # If the base field doesn't have a default, we must preserve that or provide one
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            combined_fields.append((f.name, f.type))
        else:
            combined_fields.append((f.name, f.type, f))

    combined_fields.extend(metrics_fields)

    # Create the dataclass dynamically
    DynamicState = dataclasses.make_dataclass(
        cls_name=f"Augmented_{base_cls.__name__}",
        fields=combined_fields,
    )

    # Register it with Flax struct so JAX can trace it as a PyTree
    DynamicState = struct.dataclass(DynamicState)

    _AUGMENTED_STATE_CACHE[base_cls] = DynamicState
    return DynamicState


class MetricsWrapper(object):
    """
        Logs env metrics for later use, greatly inspired by the following from PureJAXRL
        https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py
    """

    def __init__(self, wrapped_metrics_env):
        self._wrapped_metrics_env = wrapped_metrics_env

        # Probe the environment to see what specific state it uses
        _, sample_env_state = self._wrapped_metrics_env.reset(self._wrapped_metrics_env.default_params, jrandom.key(0))
        self._base_env_cls = type(sample_env_state)
        self._base_fields = {f.name for f in dataclasses.fields(self._base_env_cls)}

        # Create our flat, augmented state class specific to this sub-env
        self.StateCls = create_augmented_state_class(sample_env_state)

    @property
    def wrapped_metrics_env(self):
        return self._wrapped_metrics_env

    def _to_base_env_state(self, augmented_state) -> Any:
        """Strips the metrics fields and returns the original sub-env state class."""
        base_kwargs = {name: getattr(augmented_state, name) for name in self._base_fields}
        return self._base_env_cls(**base_kwargs)

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             action: chex.Numeric,
             state: MetricsEnvState,
             params: EnvParams,
             key: chex.PRNGKey,
             ) -> Tuple[chex.Array, chex.Array, MetricsEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        # 1. Strip metrics before passing to the underlying env step
        base_env_state = self._to_base_env_state(state)

        obs, delta_obs, nenv_state, reward, done, info = self._wrapped_metrics_env.step(
            action, base_env_state, params, key
        )

        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1

        # 2. Re-flatten the new environment state fields
        new_base_kwargs = {name: getattr(nenv_state, name) for name in self._base_fields}

        # 3. Pack everything back into our flat state
        new_state = self.StateCls(
            **new_base_kwargs,
            episode_lengths=new_episode_length * (1 - done),
            episode_returns=new_episode_return * (1 - done),
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            timestep=state.timestep + 1,
        )

        info["returned_episode_returns"] = new_state.returned_episode_returns
        info["returned_episode_lengths"] = new_state.returned_episode_lengths
        info["timestep"] = new_state.timestep
        info["returned_episode"] = done

        return obs, delta_obs, new_state, reward, done, info
        # TODO how best can I do this metrics wrapper to avoid the env_state.env_state thing that only exists when I use the metrics wrapper?
        # TODO it is a challenging problem

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, MetricsEnvState]:
        obs, base_env_state = self._wrapped_metrics_env.reset(params, key)

        base_kwargs = {name: getattr(base_env_state, name) for name in self._base_fields}

        # Build the flat state with base fields + default metric values (0 / 0.0)
        state = self.StateCls(**base_kwargs)

        return obs, state

    def __getattr__(self, attr):
        if attr == "_wrapped_metrics_env":
            raise AttributeError()
        return getattr(self._wrapped_metrics_env, attr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_metrics_env)