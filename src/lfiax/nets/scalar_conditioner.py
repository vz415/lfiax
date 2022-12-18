"""Makes scalar conditioner for conditional normalizing flow model."""

import jax.numpy as jnp
import numpy as np
import haiku as hk

from typing import Sequence


def scalar_conditioner_mlp(
    event_shape: Sequence[int],
    cond_info_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    standardize_z: bool = False,
) -> hk.Module:
    class ConditionerModule(hk.Module):
        def __call__(self, z):
            """z represents the conditioned values."""
            if standardize_z:
                z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-14)
            z = hk.Flatten(preserve_dims=-len(cond_info_shape))(z)
            z = hk.nets.MLP(hidden_sizes, activate_final=True)(z)
            z = hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            )(z)
            z = hk.Reshape(
                tuple(event_shape) + (num_bijector_params,), preserve_dims=-1
            )(z)
            return z

    return ConditionerModule()
