"""Masked conditional coupling bijector."""

from typing import Any, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.bijectors.masked_coupling import MaskedCoupling
from distrax._src.utils import math
import jax
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

    - Forward: `y = (1-m) * f(x; g(m*x|z)) + m*x`

    - Forward Jacobian log determinant:
      `log|det J(x)| = sum((1-m) * log|df/dx(x; g(m*x|z))|)`

    - Inverse: `x = (1-m) * f^{-1}(y; g(m*y|z)) + m*y`

    - Inverse Jacobian log determinant:
      `log|det J(y)| = sum((1-m) * log|df^{-1}/dy(y; g(m*y|z))|)`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_and_log_det(
        self, x: Array, theta: Array, xi: Array
    ) -> Tuple[Array, Array]:
        """Computes y = f(x|z) and log|det J(f)(x|z)|."""
        self._check_forward_input_shape(x)
        masked_x = jnp.where(self._event_mask, x, 0.0)
        # TODO: Better logic to detect when scalar x
        # if masked_x.shape[1] == 1:
        #     # breakpoint()
        #     params = self._conditioner(theta, xi)
        # else:
        #     # breakpoint()
        #     params = self._conditioner(masked_x, theta, d, xi)
        # pred = jnp.equal(masked_x.shape[1], 1)
        # params = jax.lax.cond(pred, 
        #                 lambda: self._conditioner(theta, xi),
        #                 lambda: self._conditioner(masked_x, theta, d, xi))
        params = self._conditioner(masked_x, theta, xi)
        # breakpoint()
        # jax.debug.print("params: {}", params)
        y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
        # def loss_fun(params):
        #     y, logdet = self._inner_bijector(params).forward_and_log_det(x)
        #     return -logdet
        
        # grads = jax.grad(loss_fun)(params)
        # jax.debug.print("Gradients of the loss function with respect to the conditioning network's parameters: {}", grads)
        # jax.debug.print("unmasked y: {}", y0)
        # TODO: Check that this makes sense...
        if masked_x.shape[1] > 1:
            y = jnp.where(self._event_mask, x, y0)
        else:
            y = y0
        # jax.debug.print("masked y: {}", y)
        logdet = math.sum_last(
            jnp.where(self._mask, 0.0, log_d),
            self._event_ndims - self._inner_event_ndims,
        )
        return y, logdet

    def inverse_and_log_det(
        self, y: Array, theta: Array, xi: Array
    ) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y|z) and log|det J(f^{-1})(y|z)|."""
        self._check_inverse_input_shape(y)
        masked_y = jnp.where(self._event_mask, y, 0.0)
        # TODO: Better logic to detect when scalar y
        # if masked_y.shape[1] == 1:
        #     params = self._conditioner(theta, xi)
        # else:
        #     params = self._conditioner(masked_y, theta, d, xi)
        # pred = jnp.equal(masked_y.shape[1], 1)
        # params = jax.lax.cond(pred, 
        #                 lambda: self._conditioner(theta, xi),
        #                 lambda: self._conditioner(masked_y, theta, d, xi))
        params = self._conditioner(masked_y, theta, xi)
        x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
        x = jnp.where(self._event_mask, y, x0)
        logdet = math.sum_last(
            jnp.where(self._mask, 0.0, log_d),
            self._event_ndims - self._inner_event_ndims,
        )
        return x, logdet
