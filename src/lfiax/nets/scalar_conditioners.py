"""Makes scalar conditioner for conditional normalizing flow model."""

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from typing import Sequence

# import jax.interpreters.xla as xla

# import jax.tree_util

# jax.tree_util.register_pytree_node(ConditionerModule, lambda x: (x.__class__, x.__call__))

# @jax.jit
class ScalarConditionerModule(hk.Module):
    def __init__(self, event_shape, cond_info_shape, hidden_sizes, num_bijector_params, standardize_theta=False, resnet=True):
        super(ScalarConditionerModule, self).__init__()
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
        theta = hk.Flatten()(theta)
        xi = hk.Flatten()(xi)
        z = jnp.concatenate((theta, xi), axis=1)
        if self.resnet:
            for hidden in self.hidden_sizes:
                z = hk.nets.MLP([hidden], activate_final=True)(z)
                z += hk.Linear(hidden)(hk.Flatten()(z))
        else:
            z = hk.nets.MLP(hidden_sizes, activate_final=True)(z)
        z = hk.Linear(
            np.prod(self.event_shape) * self.num_bijector_params,
            w_init=jnp.zeros,
            b_init=jnp.zeros,
        )(z)
        z = hk.Reshape(
            tuple(self.event_shape) + (self.num_bijector_params,), preserve_dims=-1
        )(z)
        return z
        


def scalar_conditioner_mlp(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    standardize_theta: bool = False,
    resnet: bool = True,
) -> hk.Module:
    class ScalarConditionerModule(hk.Module):
        def __call__(self, x, theta, xi):
            """z represents the conditioned values."""
            if standardize_theta:
                theta = (theta - theta.mean(axis=0)) / (theta.std(axis=0) + 1e-14)
            # print("in scalar")
            # breakpoint()
            theta = hk.Flatten()(theta)
            xi = hk.Flatten()(xi)
            z = jnp.concatenate((theta, xi), axis=1)  # TODO: remove x and d...?
            # breakpoint()
            # jax.debug.print("z: {}", z)
            if resnet:
                for hidden in hidden_sizes:
                    z = hk.nets.MLP([hidden], activate_final=True)(z)
                    z += hk.Linear(hidden)(hk.Flatten()(z))
            else:
                z = hk.nets.MLP(hidden_sizes, activate_final=True)(z)
            # jax.debug.print("z prev: {}", z)
            # breakpoint()
            z = hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            )(z)
            # hk.Linear(np.prod(event_shape) * num_bijector_params, w_init=jnp.zeros, b_init=jnp.zeros)(z)
            # jax.debug.print("z: {}", z)
            z = hk.Reshape(
                tuple(event_shape) + (num_bijector_params,), preserve_dims=-1
            )(z)
            # jax.debug.print("z: {}", z)
            return z

    return ScalarConditionerModule()
