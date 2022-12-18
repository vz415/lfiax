"""Conditional MNIST example. Majority of the code is reused from the original
distrax repo.
"""
from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

from lfiax.flows.nsf import make_nsf

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


# ----------------------------------------
# Helper functions to load and process data
# ----------------------------------------
def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
    ds = tfds.load("mnist", split=split, shuffle_files=True)
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
    return (
        data / 256.0,
        label,
    )  # Normalize pixel values from [0, 256) to [0, 1).


# ----------------------------
# Haiku transform functions for training and evaluation
# ----------------------------
@hk.without_apply_rng
@hk.transform
def log_prob(data: Array, cond_data: Array) -> Array:
    model = make_nsf(
        event_shape=MNIST_IMAGE_SHAPE,
        cond_info_shape=cond_info_shape,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_x=False,
        base_dist="uniform",
    )
    return model.log_prob(data, cond_data)


@hk.without_apply_rng
@hk.transform
def model_sample(key: PRNGKey, num_samples: int, cond_data: Array) -> Array:
    model = make_nsf(
        event_shape=MNIST_IMAGE_SHAPE,
        cond_info_shape=cond_info_shape,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_x=False,
        base_dist="uniform",
    )
    z = jnp.repeat(cond_data, num_samples, axis=0)
    z = jnp.expand_dims(z, -1)
    return model._sample_n(key=key, n=[num_samples], z=z)


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
def update(
    params: hk.Params, prng_key: PRNGKey, opt_state: OptState, batch: Batch
) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, prng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


if __name__ == "__main__":
    MNIST_IMAGE_SHAPE = (28, 28, 1)
    # Test code to test standardizing module
    EVENT_DIM = len(MNIST_IMAGE_SHAPE)
    cond_info_shape = (10, 1)
    batch_size = 128

    flow_num_layers = 10
    mlp_num_layers = 4
    hidden_size = 500
    num_bins = 4
    learning_rate = 1e-4

    training_steps = 1  # 000
    eval_frequency = 100

    optimizer = optax.adam(learning_rate)

    # Training
    prng_seq = hk.PRNGSequence(42)
    params = log_prob.init(
        next(prng_seq),
        np.zeros((1, *MNIST_IMAGE_SHAPE)),
        np.zeros((1, *cond_info_shape)),
    )
    opt_state = optimizer.init(params)

    train_ds = load_dataset(tfds.Split.TRAIN, batch_size)
    valid_ds = load_dataset(tfds.Split.TEST, batch_size)

    for step in range(training_steps):
        params, opt_state = update(params, next(prng_seq), opt_state, next(train_ds))

        if step % eval_frequency == 0:
            val_loss = eval_fn(params, next(valid_ds))
            print(f"STEP: {step:5d}; Validation loss: {val_loss:.3f}")
