"""Conditional chain bijector."""

from typing import Any, Callable, Optional, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors.chain import Chain
import jax.numpy as jnp


Array = base.Array
BijectorParams = Any


class ConditionalChain(Chain):
  def __init__(self, *args):
    super().__init__(*args)

  def forward(self, x: Array, z: Array) -> Array:
    """Computes y = f(x)."""
    for bijector in reversed(self._bijectors):
      x = bijector.forward(x, z)
    return x

  def inverse(self, y: Array, z: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    for bijector in self._bijectors:
      y = bijector.inverse(y, z)
    return y

  def forward_and_log_det(self, x: Array, z: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    x, log_det = self._bijectors[-1].forward_and_log_det(x, z)
    for bijector in reversed(self._bijectors[:-1]):
      x, ld = bijector.forward_and_log_det(x, z)
      log_det += ld
    return x, log_det

  def inverse_and_log_det(self, y: Array, z: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    y, log_det = self._bijectors[0].inverse_and_log_det(y, z)
    for bijector in self._bijectors[1:]:
      y, ld = bijector.inverse_and_log_det(y, z)
      log_det += ld
    return y, log_det