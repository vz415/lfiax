"""Conditional transformed distribution."""

from typing import Tuple

from distrax._src.bijectors import bijector as bjct_base
from distrax._src.distributions import distribution as dist_base
from distrax._src.distributions.transformed import Transformed
import jax
import jax.numpy as jnp


Array = dist_base.Array
DistributionLike = dist_base.DistributionLike
BijectorLike = bjct_base.BijectorLike

Array = jnp.ndarray
PRNGKey = Array


class ConditionalTransformed(Transformed):
    def __init__(self, distribution, flow):
        super().__init__(distribution, flow)

    def _sample_n(self, key: PRNGKey, n: int, theta: Array, d: Array, xi: Array) -> Array:
        """Returns `n` samples conditioned on `z`."""
        x = self.distribution.sample(seed=key, sample_shape=n)
        y, _ = self.bijector.forward_and_log_det(x, theta, d, xi)
        return y

    def log_prob(self, value: Array, theta: Array, d: Array, xi: Array) -> Array:
        """See `Distribution.log_prob`."""
        x, ildj_y = self.bijector.inverse_and_log_det(value, theta, d, xi)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def _sample_n_and_log_prob(
        self, key: PRNGKey, n: int, theta: Array, d: Array, xi: Array
    ) -> Tuple[Array, Array]:
        """Returns `n` samples and their log probs depending on `z`.

        This function is more efficient than calling `sample` and `log_prob`
        separately, because it uses only the forward methods of the bijector. It
        also works for bijectors that don't implement inverse methods.

        Args:
          key: PRNG key.
          n: Number of samples to generate.

        Returns:
          A tuple of `n` samples and their log probs.
        """
        x, lp_x = self.distribution.sample_and_log_prob(seed=key, sample_shape=n)
        y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, theta, d, xi)
        lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
        return y, lp_y
