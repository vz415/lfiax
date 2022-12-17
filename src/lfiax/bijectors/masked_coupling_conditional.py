"""Masked conditional coupling bijector."""

from typing import Any, Callable, Optional, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors.masked_coupling import MaskedCoupling
from distrax._src.utils import math
import jax.numpy as jnp


Array = base.Array
BijectorParams = Any


class MaskedConditionalCoupling(MaskedCoupling):
  """Conditional coupling bijector that uses a mask to specify which inputs 
  are transformed.

  See distrax documentation for more details about the bijector implementation.

  Let `f` be a conditional bijector (the inner bijector), `g` be a function (the
  conditioner), and `m` be a boolean mask interpreted numerically, such that
  True is 1 and False is 0. The masked coupling bijector is defined as follows:

  - Forward: `y = (1-m) * f(x; g(m*x)) + m*x`

  - Forward Jacobian log determinant:
    `log|det J(x)| = sum((1-m) * log|df/dx(x; g(m*x))|)`

  - Inverse: `x = (1-m) * f^{-1}(y; g(m*y)) + m*y`

  - Inverse Jacobian log determinant:
    `log|det J(y)| = sum((1-m) * log|df^{-1}/dy(y; g(m*y))|)`
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    
  def forward_and_log_det(self, x: Array, z: Array) -> Tuple[Array, Array]:
    """Computes y = f(x|z) and log|det J(f)(x|z)|."""
    self._check_forward_input_shape(x)
    masked_x = jnp.where(self._event_mask, x, 0.)
    # TODO: Better logic to detect when scalar x
    if masked_x.shape[1] == 1:
        params = self._conditioner(z)
    else:
        params = self._conditioner(masked_x, z)
    y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
    y = jnp.where(self._event_mask, x, y0)
    logdet = math.sum_last(
        jnp.where(self._mask, 0., log_d),
        self._event_ndims - self._inner_event_ndims)
    return y, logdet