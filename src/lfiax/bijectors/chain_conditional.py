"""Conditional chain bijector."""

from typing import Any, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors.chain import Chain


Array = base.Array
BijectorParams = Any


class ConditionalChain(Chain):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: Array, z: Array) -> Array:
        """Computes y = f(x|z)."""
        for bijector in reversed(self._bijectors):
            x = bijector.forward(x, z)
        return x

    def inverse(self, y: Array, z: Array) -> Array:
        """Computes x = f^{-1}(y|z)."""
        for bijector in self._bijectors:
            y = bijector.inverse(y, z)
        return y

    def forward_and_log_det(self, x: Array, z: Array) -> Tuple[Array, Array]:
        """Computes y = f(x|z) and log|det J(f)(x|z)|."""
        x, log_det = self._bijectors[-1].forward_and_log_det(x, z)
        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x, z)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(self, y: Array, z: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y|z) and log|det J(f^{-1})(y|z)|."""
        y, log_det = self._bijectors[0].inverse_and_log_det(y, z)
        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y, z)
            log_det += ld
        return y, log_det
