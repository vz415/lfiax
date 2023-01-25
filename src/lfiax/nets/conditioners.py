"""Makes conditioner for conditional normalizing flow model."""

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from typing import Sequence

# import jax.interpreters.xla as xla

# import jax.tree_util

# jax.tree_util.register_pytree_node(ConditionerModule, lambda x: (x.__class__, x.__call__))


class ConditionerModule(hk.Module):
    def __init__(self, event_shape, cond_info_shape, hidden_sizes, num_bijector_params, standardize_theta=False, resnet=True):
        self.event_shape = event_shape
        self.cond_info_shape = cond_info_shape
        self.hidden_sizes = hidden_sizes
        self.num_bijector_params = num_bijector_params
        self.standardize_theta = standardize_theta
        self.resnet = resnet

    def __call__(self, x, theta, d, xi):
        """x represents data and z its conditional values."""
        if self.standardize_theta:
            # Normalize the conditioned values
            theta = (theta - theta.mean(axis=0)) / (theta.std(axis=0) + 1e-14)
        x = hk.Flatten(preserve_dims=-len(self.event_shape))(x)
        theta = hk.Flatten()(theta)
        d = hk.Flatten()(d)
        xi = hk.Flatten()(xi)
        z = jnp.concatenate((theta, d, xi), axis=1)
        x = jnp.concatenate((x, z), axis=1)
        if self.resnet:
            for hidden in self.hidden_sizes:
                x = hk.nets.MLP([hidden], activate_final=True)(x)
                x += hk.Linear(hidden)(hk.Flatten()(x))
        else:
            x = hk.nets.MLP(self.hidden_sizes, activate_final=True)(x)
        x = hk.Linear(
            np.prod(self.event_shape) * self.num_bijector_params,
            w_init=jnp.zeros,
            b_init=jnp.zeros,
        )(x)
        x = hk.Reshape(
            tuple(self.event_shape) + (self.num_bijector_params,), preserve_dims=-1
        )(x)
        return x

# conditioner_module = ConditionerModule(event_shape, cond_info_shape, hidden_sizes, num_bijector_params, standardize_theta, resnet)
# conditioner_module_fn = jax.jit(conditioner_module)



def conditioner_mlp(
    event_shape: Sequence[int],
    cond_info_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    standardize_theta: bool = False,
    resnet: bool = True,
) -> hk.Module:
    class ConditionerModule(hk.Module):
        # def __call__(self, x, theta, d, xi):
        def __call__(self, x, theta, xi):
            """x represents data and z its conditional values."""
            if standardize_theta:
                # Normalize the conditioned values
                theta = (theta - theta.mean(axis=0)) / (theta.std(axis=0) + 1e-14)
            # print("in non-scalar")
            # breakpoint()
            x = hk.Flatten(preserve_dims=-len(event_shape))(x)
            theta = hk.Flatten()(theta)
            # d = hk.Flatten()(d)
            xi = hk.Flatten()(xi)
            # z = jnp.concatenate((theta, d, xi), axis=1)
            z = jnp.concatenate((theta, xi), axis=1)
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
