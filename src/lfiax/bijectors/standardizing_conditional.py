"""Conditional standardizing bijector."""

from typing import Any, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors.scalar_affine import ScalarAffine
import jax
import jax.numpy as jnp


Array = base.Array
BijectorParams = Any


class StandardizingBijector(ScalarAffine):
    """Conditional bijector to standardize inputs.

    Performs a shift and scale opeartion on input while ignoring conditioning data
    z. This is required because a composition of bijectors into a conditonal
    transformed distribution must accept conditional data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, theta: Array, d: Array, xi: Array) -> Array:
        """Computes y = f(x) = \mu + x + \sigma. Ignores z."""
        batch_shape = jax.lax.broadcast_shapes(self._batch_shape, x.shape)
        batched_scale = jnp.broadcast_to(self._scale, batch_shape)
        batched_shift = jnp.broadcast_to(self._shift, batch_shape)
        return batched_scale * x + batched_shift

    def forward_log_det_jacobian(self, x: Array, theta: Array, d: Array, xi: Array) -> Array:
        """Computes log|det J(f)(x)| ignoring z."""
        batch_shape = jax.lax.broadcast_shapes(self._batch_shape, x.shape)
        return jnp.broadcast_to(self._log_scale, batch_shape)

    def forward_and_log_det(self, x: Array, theta: Array, d: Array, xi: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)| ignoring z."""
        return self.forward(x, theta, d, xi), self.forward_log_det_jacobian(x, theta, d, xi)

    def inverse(self, y: Array, theta: Array, d: Array, xi: Array) -> Array:
        """Computes x = f^{-1}(y) ignoring z."""
        batch_shape = jax.lax.broadcast_shapes(self._batch_shape, y.shape)
        batched_inv_scale = jnp.broadcast_to(self._inv_scale, batch_shape)
        batched_shift = jnp.broadcast_to(self._shift, batch_shape)
        return batched_inv_scale * (y - batched_shift)

    def inverse_log_det_jacobian(self, y: Array, theta: Array, d: Array, xi: Array) -> Array:
        """Computes log|det J(f^{-1})(y)| ignoring z."""
        batch_shape = jax.lax.broadcast_shapes(self._batch_shape, y.shape)
        return jnp.broadcast_to(jnp.negative(self._log_scale), batch_shape)

    def inverse_and_log_det(self, y: Array, theta: Array, d: Array, xi: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)| ignoring z."""
        return self.inverse(y, theta, d, xi), self.inverse_log_det_jacobian(y, theta, d, xi)
