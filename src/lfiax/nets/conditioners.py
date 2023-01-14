"""Makes conditioner for conditional normalizing flow model."""

import jax.numpy as jnp
import numpy as np
import haiku as hk

from typing import Sequence


def conditioner_mlp(
    event_shape: Sequence[int],
    cond_info_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    standardize_z: bool = False,
    resnet: bool = False,
) -> hk.Module:
    class ConditionerModule(hk.Module):
        def __call__(self, x, z):
            """x represents data and z its conditional values."""
            if standardize_z:
                # Normalize the conditioned values
                z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-14)
            x = hk.Flatten(preserve_dims=-len(event_shape))(x)
            z = hk.Flatten(preserve_dims=-len(cond_info_shape))(z)
            x = jnp.concatenate((x, z), axis=1)
            if resnet:
                for hidden in hidden_sizes:
                    x = hk.nets.MLP([hidden], activate_final=True)(x)
                    x += hk.Linear(hidden)(hk.Flatten()(x))
            else:
                x = hk.nets.MLP(hidden_sizes, activate_final=True)(x)
            x = hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            )(x)
            x = hk.Reshape(
                tuple(event_shape) + (num_bijector_params,), preserve_dims=-1
            )(x)
            return x

    return ConditionerModule()
