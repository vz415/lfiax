import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom

import haiku as hk
import numpy as np
import tensorflow_datasets as tfds

from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)
Array = jnp.ndarray
Batch = Mapping[str, np.ndarray]

def jax_lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    if jnp.isscalar(A):
        A = A * jnp.ones(dimensions)
        return A
    shape = tuple(dimensions) + A.shape
    A = A[jnp.newaxis, ...]
    A = jnp.broadcast_to(A, shape)
    return A


def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
    """Helper function for loading and preparing tfds splits."""
    ds = split
    ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1000)
    ds = ds.repeat()
    return iter(tfds.as_numpy(ds))


def prepare_tf_dataset(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
    """Helper function for preparing tfds splits for use in fliax."""
    # TODO: add length arguments to function.
    # Batch is [y, thetas, d]
    data = batch.astype(np.float32)
    x = data[:, :len_x]
    cond_data = data[:, len_x:]
    theta = cond_data[:, :-len_x]
    d = cond_data[:, -len_x:-len_xi]
    xi = cond_data[:, -len_xi:]
    return x, theta, d, xi


@hk.without_apply_rng
@hk.transform
def log_prob(data: Array, theta: Array, xi: Array) -> Array:
    # Get batch
    shift = data.mean(axis=0)
    scale = data.std(axis=0) + 1e-14

    model = make_nsf(
        event_shape=EVENT_SHAPE,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_x=True,
        standardize_theta=False,
        use_resnet=True,
        event_dim=EVENT_DIM,
        shift=shift,
        scale=scale,
    )
    return model.log_prob(data, theta, xi)