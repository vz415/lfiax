from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Callable, Union

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

from distrax._src.bijectors import bijector as bjct_base
from distrax._src.bijectors.masked_coupling import MaskedCoupling
from distrax._src.bijectors.inverse import Inverse
from distrax._src.bijectors.chain import Chain

from distrax._src.distributions import distribution as dist_base
from distrax._src.distributions.transformed import Transformed
from distrax._src.utils import conversion, math


Array = dist_base.Array
DistributionLike = dist_base.DistributionLike
BijectorLike = bjct_base.BijectorLike

Array = bjct_base.Array
BijectorParams = Any

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


class MaskedConditionalCoupling(MaskedCoupling):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    
  def forward_and_log_det(self, x: Array, z: Array) -> Tuple[Array, Array]:
    """Computes y = f(x|z) and log|det J(f)(x|z)|."""
    self._check_forward_input_shape(x)
    masked_x = jnp.where(self._event_mask, x, 0.)
    params = self._conditioner(masked_x, z)
    y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
    y = jnp.where(self._event_mask, x, y0)
    logdet = math.sum_last(
        jnp.where(self._mask, 0., log_d),
        self._event_ndims - self._inner_event_ndims)
    return y, logdet

  def inverse_and_log_det(self, y: Array, z: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y|z) and log|det J(f^{-1})(y|z)|."""
    self._check_inverse_input_shape(y)
    masked_y = jnp.where(self._event_mask, y, 0.)
    params = self._conditioner(masked_y, z)
    x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
    x = jnp.where(self._event_mask, y, x0)
    logdet = math.sum_last(jnp.where(self._mask, 0., log_d),
                           self._event_ndims - self._inner_event_ndims)
    return x, logdet


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


class ConditionalTransformed(Transformed):
  def __init__(self, distribution, flow):
    super().__init__(distribution, flow)

  def _sample_n(self, key: PRNGKey, n: int, z: Array) -> Array:
    """Returns `n` samples conditioned on `z`."""
    x = self.distribution.sample(seed=key, sample_shape=n)
    y, _ = self.bijector.forward_and_log_det(x, z)
    return y

  def log_prob(self, value: Array, z: Array) -> Array:
    """See `Distribution.log_prob`."""
    x, ildj_y = self.bijector.inverse_and_log_det(value, z)
    lp_x = self.distribution.log_prob(x)
    lp_y = lp_x + ildj_y
    return lp_y

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int, z: Array) -> Tuple[Array, Array]:
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
    y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, z)
    lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
    return y, lp_y

# ---------------------
# Making the flow model
# --------------------- 
def make_conditioner2(event_shape: Sequence[int],
                     cond_info_shape: Sequence[int],
                     hidden_sizes: Sequence[int],
                     num_bijector_params: int) -> hk.Module: # Is this correct?
  class ConditionerModule(hk.Module):
    def __call__(self, x, z):
      x = hk.Flatten(preserve_dims=-len(event_shape))(x)
      z = hk.Flatten(preserve_dims=-len(cond_info_shape))(z)
      x = jnp.concatenate((x, z), axis=1)
      x = hk.nets.MLP(hidden_sizes, activate_final=True)(x)
      x = hk.Linear(
          np.prod(event_shape) * num_bijector_params,
          w_init=jnp.zeros,
          b_init=jnp.zeros)(x)
      x = hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1)(x)
      return x
  return ConditionerModule()

def transformer_conditioner(event_shape: Sequence[int],
                     cond_info_shape: Sequence[int],
                     hidden_sizes: Sequence[int],
                     num_bijector_params: int) -> hk.Module: # Is this correct?
  class ConditionerModule(hk.Module):
    def __call__(self, x, z):
      x = hk.Flatten(preserve_dims=-len(event_shape))(x)
      z = hk.Flatten(preserve_dims=-len(cond_info_shape))(z)
      x = jnp.concatenate((x, z), axis=1)
      x = hk.nets.MLP(hidden_sizes, activate_final=True)(x)
      x = hk.Linear(
          np.prod(event_shape) * num_bijector_params,
          w_init=jnp.zeros,
          b_init=jnp.zeros)(x)
      x = hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1)(x)
      return x
  return ConditionerModule()


def make_flow_model(event_shape: Sequence[int],
                    cond_info_shape: Sequence[int],
                    num_layers: int,
                    hidden_sizes: Sequence[int],
                    num_bins: int) -> distrax.Transformed:
  """Creates the flow model."""
  # Alternating binary mask.
  mask = jnp.arange(0, np.prod(event_shape)) % 2
  mask = jnp.reshape(mask, event_shape)
  mask = mask.astype(bool)

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=0., range_max=1.)

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.
  num_bijector_params = 3 * num_bins + 1

  layers = []
  for _ in range(num_layers):
    # layer = distrax.MaskedCoupling(
    layer = MaskedConditionalCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=make_conditioner2(event_shape, 
                                     cond_info_shape,
                                     hidden_sizes,
                                     num_bijector_params))
    layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # We invert the flow so that the `forward` method is called with `log_prob`.
  # flow = distrax.Inverse(distrax.Chain(layers))
  breakpoint()
  flow = ConditionalInverse(ConditionalChain(layers))
  base_distribution = distrax.Independent(
      distrax.Uniform(
          low=jnp.zeros(event_shape),
          high=jnp.ones(event_shape)),
      reinterpreted_batch_ndims=len(event_shape))

  # return distrax.Transformed(base_distribution, flow)
  return ConditionalTransformed(base_distribution, flow)


# ----------------------------------------
# Helper functions to load and process data
# ----------------------------------------
def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
  ds = tfds.load("mnist", split=split, shuffle_files=True)
  # ds = split
  ds = ds.shuffle(buffer_size=10 * batch_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=1000)
  ds = ds.repeat()
  return iter(tfds.as_numpy(ds))


def one_hot_mnist(x, dtype=jnp.float32):
  """Create a one-hot encoding of x of size 10 for MNIST."""
  return jnp.array(x[:, None] == jnp.arange(10), dtype)


def prepare_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
  data = batch["image"].astype(np.float32)
  label = batch["label"].astype(np.float32)
  label = one_hot_mnist(label)
  label = jnp.expand_dims(label, -1)
  if prng_key is not None:
    # Dequantize pixel values {0, 1, ..., 255} with uniform noise [0, 1).
    data += jax.random.uniform(prng_key, data.shape)
  return data / 256., label  # Normalize pixel values from [0, 256) to [0, 1).


# ----------------------------
# Haiku transform functions for training and evaluation
# ----------------------------
@hk.without_apply_rng
@hk.transform
def log_prob(data: Array, cond_data: Array) -> Array:
  model = make_flow_model(
      event_shape=MNIST_IMAGE_SHAPE,
      cond_info_shape=cond_info_shape,
      num_layers=flow_num_layers,
      hidden_sizes=[hidden_size] * mlp_num_layers,
      num_bins=num_bins)
  return model.log_prob(data, cond_data)

@hk.without_apply_rng
@hk.transform
def model_sample(key: PRNGKey, num_samples: int, cond_data: Array) -> Array:
  model = make_flow_model(
      event_shape=MNIST_IMAGE_SHAPE,
      cond_info_shape=cond_info_shape,
      num_layers=flow_num_layers,
      hidden_sizes=[hidden_size] * mlp_num_layers,
      num_bins=num_bins)
  z = jnp.repeat(cond_data, num_samples, axis=0)
  z = jnp.expand_dims(z, -1)
  return model._sample_n(key=key, 
                         n=[num_samples],
                         z=z)

def loss_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch) -> Array:
  data = prepare_data(batch, prng_key)
  # Loss is average negative log likelihood.
  loss = -jnp.mean(log_prob.apply(params, data[0], data[1]))
  return loss

@jax.jit
def eval_fn(params: hk.Params, batch: Batch) -> Array:
  data = prepare_data(batch)  # We don't dequantize during evaluation.
  loss = -jnp.mean(log_prob.apply(params, data[0], data[1]))
  return loss

@jax.jit
def update(params: hk.Params,
            prng_key: PRNGKey,
            opt_state: OptState,
            batch: Batch) -> Tuple[hk.Params, OptState]:
  """Single SGD update step."""
  grads = jax.grad(loss_fn)(params, prng_key, batch)
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return new_params, new_opt_state


MNIST_IMAGE_SHAPE = (28, 28, 1)
cond_info_shape = (10,1)
batch_size = 128

flow_num_layers = 10
mlp_num_layers = 4
hidden_size = 500
num_bins = 4
learning_rate = 1e-4

# using 100,000 steps could take long (about 2 hours) but will give better results. 
# You can try with 10,000 steps to run it fast but result may not be very good

training_steps =  5000
eval_frequency =  100

optimizer = optax.adam(learning_rate)

# Training
prng_seq = hk.PRNGSequence(42)
params = log_prob.init(next(prng_seq), 
                    np.zeros((1, *MNIST_IMAGE_SHAPE)), 
                    np.zeros((1, *cond_info_shape)))
opt_state = optimizer.init(params)

train_ds = load_dataset(tfds.Split.TRAIN, batch_size)
valid_ds = load_dataset(tfds.Split.TEST, batch_size)

for step in range(training_steps):
  params, opt_state = update(params, next(prng_seq), opt_state,
                              next(train_ds))

  if step % eval_frequency == 0:
    val_loss = eval_fn(params, next(valid_ds))
    print(f"STEP: {step:5d}; Validation loss: {val_loss:.3f}")


