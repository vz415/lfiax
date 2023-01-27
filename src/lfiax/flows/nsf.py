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

# from lfiax.nets.scalar_conditioners import scalar_conditioner_mlp
from lfiax.nets.scalar_conditioners import scalar_conditioner_mlp, ScalarConditionerModule
# from lfiax.nets.conditioners import conditioner_mlp
from lfiax.nets.conditioners import conditioner_mlp, ConditionerModule


Array = jnp.ndarray
PRNGKey = Array


def make_nsf(
    event_shape: Sequence[int],
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    standardize_x: bool,
    standardize_theta: bool = False,
    use_resnet: bool = True,
    event_dim: int = None,
    shift: float = None,
    scale: float = None,
    base_dist: str = "gaussian",
) -> distrax.Transformed:
    """Creates a neural spline flow (nsf) model using conditional
    bijectors and distributions. Heavily inspired/copied off
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

    # Starting with an inverted standardizing layer - could put non-inverted
    # standardizing layer at end of `layers` list
    if standardize_x:
        layers = [
            ConditionalInverse(
                ConditionalBlock(StandardizingBijector(shift, scale), event_dim)
            )
        ]
    else:
        layers = []
    # self._conditioner.params_dict()
    # @jax.jit
    # def scalar_conditioner_fn(*args, **kwargs):
    #     return scalar_conditioner_mlp(*args, **kwargs)

    # @jax.jit
    # def non_scalar_conditioner_fn(*args, **kwargs):
        # return conditioner_mlp(*args, **kwargs)
    # breakpoint()
    # conditioner = jax.lax.cond(
    #     event_shape == (1,), 
    #     jax.jit(lambda a, b, c, d, e, f: ScalarConditionerModule(a, b, c, d, e, f)),
    #     ConditionerModule, 
    #     event_shape, cond_info_shape, hidden_sizes, num_bijector_params, standardize_theta, use_resnet)

    if event_shape == (1,):
        conditioner = scalar_conditioner_mlp(
            event_shape,
            # cond_info_shape,
            hidden_sizes,
            num_bijector_params,
            standardize_theta,
            use_resnet,
        )
    else:
        conditioner = conditioner_mlp(
            event_shape,
            # cond_info_shape,
            hidden_sizes,
            num_bijector_params,
            standardize_theta,
            use_resnet,
        )

    # Append subsequent layers
    for _ in range(num_layers):
        layer = MaskedConditionalCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=conditioner,
        )
        layers.append(layer)
        # Flip the mask after each layer as long as event is non-scalar.
        if event_shape != (1,):
            mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
    flow = ConditionalInverse(ConditionalChain(layers))

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
        raise AssertionError("Specified non-implemented distribution.")

    return ConditionalTransformed(base_distribution, flow)



# def cond_fn(event_shape, cond_info_shape, hidden_sizes, num_bijector_params, standardize_theta, use_resnet):
#     if event_shape == (1,):
#         return scalar_conditioner_mlp(
#             event_shape,
#             cond_info_shape,
#             hidden_sizes,
#             num_bijector_params,
#             standardize_theta,
#             use_resnet,
#         )
#     else:
#         return conditioner_mlp(
#             event_shape,
#             cond_info_shape,
#             hidden_sizes,
#             num_bijector_params,
#             standardize_theta,
#             use_resnet,
#         )

# create a function to use with jax.lax.cond
