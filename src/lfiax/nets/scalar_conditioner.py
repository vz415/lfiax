'''Makes scalar conditioner for conditional normalizing flow model.'''
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
import numpy as np
import haiku as hk

from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Callable, Union


def scalar_conditioner_mlp(event_shape: Sequence[int],
                      cond_info_shape: Sequence[int],
                      hidden_sizes: Sequence[int],
                      num_bijector_params: int) -> hk.Module: # Is this correct?
  class ConditionerModule(hk.Module):
    def __call__(self, x):
      # thetas = (thetas - thetas.mean(axis=0)) / thetas.std(axis=0)
      x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-14)
      x = hk.Flatten(preserve_dims=-len(cond_info_shape))(x)
      x = hk.nets.MLP(hidden_sizes, activate_final=True)(x)
      x = hk.Linear(
          np.prod(event_shape) * num_bijector_params, # This event_shape can be an int or tuple
          w_init=jnp.zeros,
          b_init=jnp.zeros)(x)
      # This reshape function is important for getting parameters in place for
      # subsequent evaluation by the _inner_bijector.
      
      x = hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1)(x)
      # This ^ event_shape must be a tuple...
    #   x = hk.Reshape(tuple(3) + (num_bijector_params,), preserve_dims=-1)(x)
      return x
  return ConditionerModule()