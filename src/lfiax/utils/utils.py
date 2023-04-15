import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

import haiku as hk
import numpy as np
from scipy.stats import binom
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
    """[Legacy] Helper function for preparing tfds splits for use in fliax."""
    # TODO: add length arguments to function.
    # Batch is [y, thetas, d]
    data = batch.astype(np.float32)
    x = data[:, :len_x]
    cond_data = data[:, len_x:]
    theta = cond_data[:, :-len_x]
    d = cond_data[:, -len_x:-len_xi]
    xi = cond_data[:, -len_xi:]
    return x, theta, d, xi


def sbc(prior, posterior, simulator, num_simulations=1_000, num_data_points=1_000, num_bins=None):
    """
    Compute the simulation-based calibration (SBC) histogram for the given prior and posterior
    distributions.

    It's on the user to ensure that the right number of bins are used for plotting.

    Args:
        prior: A distribution object with sample and log_prob methods representing the prior distribution.
        posterior: A distribution object with sample and log_prob methods representing the posterior distribution.
        simulator: Callable object that produces an output. Can be replaced with explicit likelihood.
        num_simulations: An integer representing the number of simulations to perform for the SBC (default: 1000).
        num_bins: An optional integer representing the number of bins in the histogram. 
                  If not provided, it will be calculated as num_simulations // 40. Represents
                  the number of simulations for each posterior and ranks to check.

    Returns:
        sbc_histogram: A NumPy array representing the SBC histogram of rank statistics.
    """
    if num_bins is None:
        num_bins = num_data_points // 40

    def sbc_iteration(key):
        # True theta values to use
        theta_true = prior.sample(seed=key)
        # likelihood given the prior
        # TODO: for both of the following, add more contextual info (e.g. theta & xi)
        x_o = simulator(theta_true)
        # posterior given observed data
        posteriors = posterior.sample(x_o, sample_shape=(num_data_points,))

        # Rank of the sample
        rank = jnp.sum(posteriors < theta_true)

        # Scale the rank values to match the reduced number of bins
        # rank_scaled = jnp.floor(rank * num_bins / num_simulations)

        return rank

    key = jrandom.PRNGKey(42)
    keys = jrandom.split(key, num_simulations)
    sbc_ranks = jax.vmap(sbc_iteration)(keys)

    # Calculate the histogram using numpy.histogram
    sbc_histogram, _ = np.histogram(sbc_ranks, bins=np.arange(num_bins + 1))
    
    return sbc_histogram


def ks_test(sample1, sample2):
    '''Two-sample KS-test.'''
    sample1_sorted = jnp.sort(sample1)
    sample2_sorted = jnp.sort(sample2)
    sample1_size = sample1.shape[0]
    sample2_size = sample2.shape[0]
    
    data_all = jnp.concatenate([sample1_sorted, sample2_sorted])
    group_indicator = jnp.concatenate([jnp.zeros(sample1_size), jnp.ones(sample2_size)])
    index_sorted = jnp.argsort(data_all)
    
    group_sorted = group_indicator[index_sorted]
    d_plus = jnp.where(group_sorted == 1, 1 / sample2_size, 0)
    d_minus = jnp.where(group_sorted == 0, 1 / sample1_size, 0)
    
    cdf_diff = jnp.cumsum(d_plus - d_minus)
    ks_statistic = jnp.max(jnp.abs(cdf_diff))
    
    # Compute p-value using asymptotic distribution
    n = sample1_size * sample2_size / (sample1_size + sample2_size)
    p_value = np.exp(-2 * n * ks_statistic**2)
    
    return ks_statistic, p_value

def c2st_accuracy(ranks: jnp.ndarray, uniforms: jnp.ndarray, num_folds: int = 5) -> float:
    """
    Perform the Classifier 2-Sample Test (C2ST) using a logistic regression classifier and
    compute the average cross-validated accuracy.

    Args:
        ranks: A JAX numpy array containing the first set of data samples (ranks in SBC).
        uniforms: A JAX numpy array containing the second set of data samples (uniform samples in SBC).
        num_folds: The number of folds to use for cross-validation (default is 5).

    Returns:
        The average cross-validated accuracy of the classifier on the two datasets.
    """
    # Combine the data and create labels
    data_combined = jnp.concatenate([ranks, uniforms])[:, None]
    labels = jnp.concatenate([jnp.zeros(ranks.shape[0]), jnp.ones(uniforms.shape[0])])

    # Define logistic regression model using Haiku
    def logistic_regression_fn(x):
        return hk.Sequential([hk.Linear(1), jax.nn.sigmoid])(x)

    logistic_regression = hk.without_apply_rng(hk.transform(logistic_regression_fn))

    # Cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True)
    accuracy_scores = []

    for train_indices, val_indices in kf.split(data_combined):
        # Split the data into training and validation sets
        x_train, x_val = data_combined[train_indices], data_combined[val_indices]
        y_train, y_val = labels[train_indices], labels[val_indices]

        # Initialize parameters
        params = logistic_regression.init(jrandom.PRNGKey(42), x_train)

        # Define the loss function
        def loss_fn(params, x, y):
            logits = logistic_regression.apply(params, x)
            return -jnp.mean(y * jnp.log(logits) + (1 - y) * jnp.log(1 - logits))

        # Define the gradient function
        grad_fn = jax.value_and_grad(loss_fn)

        # Define the optimizer
        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        # Train the logistic regression model
        for _ in range(500):
            loss, grads = grad_fn(params, x_train, y_train)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        # Compute accuracy on the validation set
        logits_val = logistic_regression.apply(params, x_val)
        preds_val = jnp.round(logits_val).squeeze()
        accuracy = jnp.mean(preds_val == y_val)
        accuracy_scores.append(accuracy)

    return jnp.mean(jnp.array(accuracy_scores))


def expected_coverage_probability(sbc_ranks: jnp.ndarray, alpha: float) -> float:
    """
    Calculate the Expected Coverage Probability (ECP) for a given value of alpha.

    Args:
        sbc_ranks: A JAX numpy array containing the SBC ranks.
        alpha: A float value between 0 and 1.

    Returns:
        The Expected Coverage Probability (ECP) as a float.
    """
    num_simulations = sbc_ranks.shape[0]
    num_ranks_exceeding_alpha = jnp.sum(sbc_ranks / num_simulations >= alpha)
    ecp = num_ranks_exceeding_alpha / num_simulations
    return ecp


