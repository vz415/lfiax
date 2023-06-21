"""Makes Neural Spline Flow normalizing flow."""

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import distrax

from lfiax.bijectors.chain_conditional import ConditionalChain
from lfiax.bijectors.block_conditional import ConditionalBlock
from lfiax.bijectors.inverse_conditional import ConditionalInverse
from lfiax.bijectors.masked_coupling_conditional import MaskedConditionalCoupling
from lfiax.bijectors.standardizing_conditional import StandardizingBijector

from lfiax.distributions.transformed_conditional import ConditionalTransformed

from lfiax.nets.scalar_conditioners import scalar_conditioner_mlp, conditional_scalar_conditioner_mlp
from lfiax.nets.conditioners import conditioner_mlp, conditional_conditioner_mlp


Array = jnp.ndarray
PRNGKey = Array


def make_nsf(
    event_shape: Sequence[int],
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    standardize_theta: bool = False,
    use_resnet: bool = True,
    conditional: bool = True,
    base_dist: str = "gaussian",
) -> distrax.Transformed:
    """Creates a neural spline flow (nsf) model using conditional or non-conditional
    bijectors and distributions given whether specified. Heavily inspired/copied off
    of the original nsf model in the distrax repo."""
    # Alternating binary mask.
    mask = jnp.arange(0, np.prod(event_shape)) % 2
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)
    if event_shape == (1,):
        mask = jnp.array([1]).astype(bool)

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

    # Number of parameters for the rational-quadratic spline:
    # - `num_bins` bin widths
    # - `num_bins` bin heights
    # - `num_bins + 1` knot slopes``
    # for a total of `3 * num_bins + 1` parameters.
    num_bijector_params = 3 * num_bins + 1

    def create_conditioner():
        """Creates a conditioner based on the given settings."""
        if event_shape == (1,):
            if conditional:
                return conditional_scalar_conditioner_mlp(
                    event_shape,
                    hidden_sizes,
                    num_bijector_params,
                    standardize_theta,
                    use_resnet,
                )
            else:
                return scalar_conditioner_mlp(
                    event_shape,
                    hidden_sizes,
                    num_bijector_params,
                    use_resnet,
                )
        else:
            if conditional:
                return conditional_conditioner_mlp(
                    event_shape,
                    hidden_sizes,
                    num_bijector_params,
                    standardize_theta,
                    use_resnet,
                )
            else:
                return conditioner_mlp(
                    event_shape,
                    hidden_sizes,
                    num_bijector_params,
                    standardize_theta,
                    use_resnet,
                )
    
    layers = []

    # Append subsequent layers
    # TODO: Make flow layer creation dependent on whether conditional input...
    if conditional:
        for _ in range(num_layers):
            layer = MaskedConditionalCoupling(
                mask=mask,
                bijector=bijector_fn,
                conditioner=create_conditioner(),
            )
            layers.append(layer)
            # Flip the mask after each layer as long as event is non-scalar.
            if event_shape != (1,):
                mask = jnp.logical_not(mask)

        # We invert the flow so that the `forward` method is called with `log_prob`.
        flow = ConditionalInverse(ConditionalChain(layers))
    else:
        for _ in range(num_layers):
            layer = distrax.MaskedCoupling(
                mask=mask,
                bijector=bijector_fn,
                conditioner=conditioner
            )
            layers.append(layer)
            # Flip the mask after each layer as long as event is non-scalar.
            if event_shape != (1,):
                mask = jnp.logical_not(mask)

        # We invert the flow so that the `forward` method is called with `log_prob`.
        flow = distrax.Inverse(distrax.Chain(layers))


    if base_dist == "gaussian":
        mu = jnp.zeros(event_shape)
        sigma = jnp.ones(event_shape)
        base_distribution = distrax.Independent(
            distrax.MultivariateNormalDiag(mu, sigma)
        )
    elif base_dist == "uniform":
        base_distribution = distrax.Independent(
            distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
            reinterpreted_batch_ndims=len(event_shape),
        )
    else:
        raise AssertionError("Specified non-implemented base distribution.")

    if conditional:
        return ConditionalTransformed(base_distribution, flow)
    else:
        return distrax.Transformed(base_distribution, flow)
