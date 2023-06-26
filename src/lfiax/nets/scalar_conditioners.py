"""Makes scalar conditioner for conditional normalizing flow model."""

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from typing import Sequence


def scalar_conditioner_mlp(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    resnet: bool = True,
) -> hk.Module:
    class ScalarConditionerModule(hk.Module):
        def __call__(self, x):
            """z represents the conditioned values."""
            x = hk.Flatten()(x)
            if resnet:
                for i, hidden in enumerate(hidden_sizes):
                    x_temp = hk.nets.MLP([hidden], activate_final=True)(x)
                    if i > 0:
                        x += x_temp
                        x = hk.LayerNorm(
                            axis=-1, create_scale=True, create_offset=True
                        )(x)
                    else:
                        x = x_temp
                        x = hk.LayerNorm(
                            axis=-1, create_scale=True, create_offset=True
                        )(x)
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

    return ScalarConditionerModule()


def conditional_scalar_conditioner_mlp(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    standardize_theta: bool = True,
    resnet: bool = True,
) -> hk.Module:
    class ScalarConditionerModule(hk.Module):
        def __call__(self, x, theta, xi):
            """z represents the conditioned values."""
            if standardize_theta:
                theta = jnp.divide(
                    jnp.subtract(theta, jnp.mean(theta, axis=0)),
                    jnp.std(theta, axis=0) + 1e-10,
                )
                # TODO: implement BatchNorm & dropout by creating stateful function
                # theta = hk.BatchNorm(theta)
            theta = hk.Flatten()(theta)
            x = hk.Flatten()(x)
            xi = hk.Flatten()(xi)
            z = jnp.concatenate((x, theta, xi), axis=1)

            if resnet:
                for i, hidden in enumerate(hidden_sizes):
                    z_temp = hk.nets.MLP([hidden], activate_final=True)(z)
                    if i > 0:
                        z += z_temp
                        z = hk.LayerNorm(
                            axis=-1, create_scale=True, create_offset=True
                        )(z)
                    else:
                        z = z_temp
                        z = hk.LayerNorm(
                            axis=-1, create_scale=True, create_offset=True
                        )(z)
            else:
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

    return ScalarConditionerModule()
