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
    # shift: float,
    # scale: float,
    standardize_theta: bool = False,
    resnet: bool = True,
) -> hk.Module:
    class ConditionerModule(hk.Module):
        def __call__(self, x, theta, xi):
            """x represents data and z its conditional values."""
            def standard_scale_nonzero_columns_inplace(x):
                # Create a boolean mask array representing non-zero columns
                mask = (x != 0)

                # Compute the mean and std for each column, only considering non-zero elements
                column_sum = jnp.sum(x * mask, axis=0)
                column_count = jnp.sum(mask, axis=0)
                column_mean = jnp.where(column_count > 0, column_sum / column_count, 0)
                column_std = jnp.sqrt(
                    jnp.where(
                        column_count > 1, 
                        jnp.sum(((x - column_mean) * mask) ** 2, axis=0) / (column_count - 1), 1))

                # Compute standardized_x by subtracting the mean and dividing by the standard deviation for non-zero elements
                standardized_x = x - mask * column_mean
                standardized_x = jnp.where(mask, standardized_x / (column_std + 1e-8), standardized_x)

                return standardized_x
                
            if standardize_theta:
                # Normalize the conditioned values
                theta = jnp.divide(
                    jnp.subtract(theta, jnp.mean(theta, axis=0)), 
                    jnp.std(theta, axis=0) + 1e-10
                    )
                # TODO: Move to stateful transform to use this ewma stateful function
                # theta = hk.BatchNorm(
                #     create_scale=True, 
                #     create_offset=True,
                #     # Never really going to "test" this
                #     decay_rate=0.99)(theta, is_training=True)
            
            x = standard_scale_nonzero_columns_inplace(x)
            
            x = hk.Flatten(preserve_dims=-len(event_shape))(x)
            theta = hk.Flatten()(theta)
            xi = hk.Flatten()(xi)
            z = jnp.concatenate((theta, xi), axis=1)
            x = jnp.concatenate((x, z), axis=1)
            if resnet:
                for i, hidden in enumerate(hidden_sizes):
                    x_temp = hk.nets.MLP(
                        [hidden],
                        w_init=hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal"),
                        activate_final=True)(x)
                    if i > 0: 
                        x += x_temp
                        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
                    else: 
                        x = x_temp
                        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            else:
                x = hk.nets.MLP(
                    hidden_sizes,
                    w_init=hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal"),
                    activate_final=True)(x)
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
