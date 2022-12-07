'''Makes Neural Spline Flow normalizing flow.'''

from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Callable, Union

import jax.numpy as jnp
import numpy as np
import distrax

from lfiax.bijectors.chain_conditional import ConditionalChain
from lfiax.bijectors.block_conditional import ConditionalBlock
from lfiax.bijectors.inverse_conditional import ConditionalInverse
from lfiax.bijectors.masked_coupling_conditional import MaskedConditionalCoupling
from lfiax.bijectors.standardizing_conditional import StandardizingBijector

from lfiax.distributions.transformed_conditional import ConditionalTransformed

from lfiax.nets.scalar_conditioner import scalar_conditioner_mlp
from lfiax.nets.conditioner import conditioner_mlp


Array = jnp.ndarray
PRNGKey = Array

def make_nsf(event_shape: Sequence[int],
              cond_info_shape: Sequence[int],
              num_layers: int,
              hidden_sizes: Sequence[int],
              num_bins: int,
              # conditioner: Callable,
              shift: float,
              scale: float,) -> distrax.Transformed:
  """Creates the flow model."""
  # Alternating binary mask.
  mask = jnp.arange(0, np.prod(event_shape)) % 2
  mask = jnp.reshape(mask, event_shape)
  mask = mask.astype(bool)
  if event_shape == (1,):
    mask = jnp.array([1]).astype(bool)

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=0., range_max=1.)

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes``
  # for a total of `3 * num_bins + 1` parameters.
  num_bijector_params = 3 * num_bins + 1

  # Starting with a standardizing layer - might want to put at end
  layers = [
      # ConditionalInverse(ConditionalBlock(StandardizingBijector(shift, scale), event_shape[0]))
      ConditionalInverse(ConditionalBlock(StandardizingBijector(shift, scale), 1))
  ]
  
  if event_shape == (1,):
    # TODO: Make customizable implementation for scalar
    conditioner = scalar_conditioner_mlp(event_shape,
                                      cond_info_shape,
                                      hidden_sizes,
                                      num_bijector_params)
  else:
    # TODO: Make customizable implementation for non-scalar
    conditioner = conditioner_mlp(event_shape,
                                      cond_info_shape,
                                      hidden_sizes,
                                      num_bijector_params)

  # Append subsequent layers
  for _ in range(num_layers):
    # Could use better logic on what type of conditioner to make based on event_shape.
    layer = MaskedConditionalCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=conditioner,)
        # event_ndims=event_shape[0])
    layers.append(layer)
    # Flip the mask after each layer.
    if event_shape != (1,):
        mask = jnp.logical_not(mask)
  
  # We invert the flow so that the `forward` method is called with `log_prob`.
  # flow = distrax.Inverse(distrax.Chain(layers))
  flow = ConditionalInverse(ConditionalChain(layers))
  
  # Making base a Gaussian.
  mu = jnp.zeros(event_shape)
  sigma = jnp.ones(event_shape)
  # For scalar, doesn't matter if Independent is used but maybe will for 2D+
  base_distribution = distrax.Independent(
      distrax.MultivariateNormalDiag(mu, sigma))
  # base_distribution = distrax.MultivariateNormalDiag(mu, sigma)
  
  # return distrax.Transformed(base_distribution, flow)
  return ConditionalTransformed(base_distribution, flow)