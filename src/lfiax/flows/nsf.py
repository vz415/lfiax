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
              standardize: bool,
              shift: float = None,
              scale: float = None,
              base_dist: str = 'gaussian',
              # embedding_net: nn.Module = nn.Identity(),
              ) -> distrax.Transformed:
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
  if standardize:
    layers = [
      # TODO: Auto-select dimensionality of ConditonalBlock - only works for 1D linear regression
      ConditionalInverse(ConditionalBlock(StandardizingBijector(shift, scale), 1))
    ]
  else:
    layers = []
  
  if event_shape == (1,):
    # TODO: assert conditioner is a scalar
    conditioner = scalar_conditioner_mlp(event_shape,
                                      cond_info_shape,
                                      hidden_sizes,
                                      num_bijector_params)
  else:
    # TODO: assert conditioner is non-scalar
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
    # Flip the mask after each layer as long as non-scalar.
    if event_shape != (1,):
        mask = jnp.logical_not(mask)
  
  # We invert the flow so that the `forward` method is called with `log_prob`.
  flow = ConditionalInverse(ConditionalChain(layers))
  
  # TODO: Make base distribution customizable
  if base_dist == 'gaussian':
    # Making base a Gaussian.
    mu = jnp.zeros(event_shape)
    sigma = jnp.ones(event_shape)
    base_distribution = distrax.Independent(
        distrax.MultivariateNormalDiag(mu, sigma))
  elif base_dist == 'uniform':
    base_distribution = distrax.Independent(
        distrax.Uniform(
            low=jnp.zeros(event_shape),
            high=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape))
  # TODO: Make specification of distributions more customizable
  else: raise AssertionError('Specified non-implemented distribution')
  
  return ConditionalTransformed(base_distribution, flow)