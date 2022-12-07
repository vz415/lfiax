"""Conditional block bijector."""

from typing import Any, Callable, Optional, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors.block import Block
# from distrax._src.utils import conversion
from distrax._src.utils import math
import jax.numpy as jnp


Array = base.Array
BijectorParams = Any


class ConditionalBlock(Block):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, x: Array, z: Array) -> Array:
    """Computes y = f(x)."""
    self._check_forward_input_shape(x)
    return self._bijector.forward(x, z)

  def inverse(self, y: Array, z: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    self._check_inverse_input_shape(y)
    return self._bijector.inverse(y, z)

  def forward_log_det_jacobian(self, x: Array, z: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    log_det = self._bijector.forward_log_det_jacobian(x, z)
    return math.sum_last(log_det, self._ndims)

  def inverse_log_det_jacobian(self, y: Array, z: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    log_det = self._bijector.inverse_log_det_jacobian(y, z)
    return math.sum_last(log_det, self._ndims)

  def forward_and_log_det(self, x: Array, z: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    y, log_det = self._bijector.forward_and_log_det(x, z)
    return y, math.sum_last(log_det, self._ndims)

  def inverse_and_log_det(self, y: Array, z: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    x, log_det = self._bijector.inverse_and_log_det(y, z)
    return x, math.sum_last(log_det, self._ndims)