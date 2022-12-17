"""Conditional inverse bijector."""

from typing import Any, Callable, Optional, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors.inverse import Inverse
import jax.numpy as jnp


Array = base.Array
BijectorParams = Any


class ConditionalInverse(Inverse):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, x: Array, z: Array) -> Array:
    """Computes y = f(x)."""
    return self._bijector.inverse(x, z)

  def inverse(self, y: Array, z: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    return self._bijector.forward(y, z)
  
  def forward_and_log_det(self, x: Array, z: Array) -> Tuple[Array, Array]:
    """Computes y = f(x|z) and log|det J(f)(x|z)|."""
    return self._bijector.inverse_and_log_det(x, z)

  def inverse_and_log_det(self, y: Array, z: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self._bijector.forward_and_log_det(y, z)