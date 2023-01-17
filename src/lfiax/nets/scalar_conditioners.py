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
    standardize_theta: bool = False,
    resnet: bool = True,
) -> hk.Module:
    class ConditionerModule(hk.Module):
        def __call__(self, theta, d, xi):
            """z represents the conditioned values."""
            if standardize_theta:
                theta = (theta - theta.mean(axis=0)) / (theta.std(axis=0) + 1e-14)
            # Original cond_info_shape = (theta_shape[0] + len(d),)
            # Now it's cond_info_shape = (theta_shape, len(d_obs), len_xi)
            theta = hk.Flatten(preserve_dims=-cond_info_shape[0])(theta)
            # d = hk.Flatten(preserve_dims=-cond_info_shape[1])(d)
            # xi = hk.Flatten(preserve_dims=-cond_info_shape[2])(xi)
            d = hk.Flatten()(d)
            xi = hk.Flatten()(xi)
            # Will this give me the same problem as before i.e. nans?
            z = jnp.concatenate((theta, d, xi), axis=1)
            if resnet:
                for hidden in hidden_sizes:
                    z = hk.nets.MLP([hidden], activate_final=True)(z)
                    z += hk.Linear(hidden)(hk.Flatten()(z))
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

    return ConditionerModule()
