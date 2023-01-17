"""Conditional block bijector."""

from typing import Any, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors.block import Block
from distrax._src.utils import math


Array = base.Array
BijectorParams = Any


class ConditionalBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, theta: Array, d: Array, xi: Array) -> Array:
        """Computes y = f(x|z)."""
        self._check_forward_input_shape(x)
        return self._bijector.forward(x, theta, d, xi)

    def inverse(self, y: Array, theta: Array, d: Array, xi: Array) -> Array:
        """Computes x = f^{-1}(y|z)."""
        self._check_inverse_input_shape(y)
        return self._bijector.inverse(y, theta, d, xi)

    def forward_log_det_jacobian(
        self, x: Array, theta: Array, d: Array, xi: Array
    ) -> Array:
        """Computes log|det J(f)(x|z)|."""
        self._check_forward_input_shape(x)
        log_det = self._bijector.forward_log_det_jacobian(x, theta, d, xi)
        return math.sum_last(log_det, self._ndims)

    def inverse_log_det_jacobian(
        self, y: Array, theta: Array, d: Array, xi: Array
    ) -> Array:
        """Computes log|det J(f^{-1})(y|z)|."""
        self._check_inverse_input_shape(y)
        log_det = self._bijector.inverse_log_det_jacobian(y, theta, d, xi)
        return math.sum_last(log_det, self._ndims)

    def forward_and_log_det(
        self, x: Array, theta: Array, d: Array, xi: Array
    ) -> Tuple[Array, Array]:
        """Computes y = f(x|z) and log|det J(f)(x|z)|."""
        self._check_forward_input_shape(x)
        y, log_det = self._bijector.forward_and_log_det(x, theta, d, xi)
        return y, math.sum_last(log_det, self._ndims)

    def inverse_and_log_det(
        self, y: Array, theta: Array, d: Array, xi: Array
    ) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y|z) and log|det J(f^{-1})(y|z)|."""
        self._check_inverse_input_shape(y)
        x, log_det = self._bijector.inverse_and_log_det(y, theta, d, xi)
        return x, math.sum_last(log_det, self._ndims)
