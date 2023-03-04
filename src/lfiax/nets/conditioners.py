"""Makes conditioner for conditional normalizing flow model."""

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from typing import Sequence


def conditioner_mlp(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    standardize_theta: bool = False,
    resnet: bool = True,
) -> hk.Module:
    class ConditionerModule(hk.Module):
        def __call__(self, x, theta, xi):
            """x represents data and z its conditional values."""
            if standardize_theta:
                # Normalize the conditioned values
                theta = hk.BatchNorm(theta)
            x = hk.Flatten(preserve_dims=-len(event_shape))(x)
            theta = hk.Flatten()(theta)
            xi = hk.Flatten()(xi)
            z = jnp.concatenate((theta, xi), axis=1)
            x = jnp.concatenate((x, z), axis=1)
            if resnet:
                for i, hidden in enumerate(hidden_sizes):
                    x_temp = hk.nets.MLP([hidden], activate_final=True)(x)
                    if i > 0: 
                        x += x_temp
                        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
                    else: 
                        x = x_temp
                        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            else:
                x = hk.nets.MLP(hidden_sizes, activate_final=True)(x)
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
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
