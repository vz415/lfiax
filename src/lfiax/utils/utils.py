import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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
PRNGKey = Array


def plot_contour_prior_posterior(prior_samples, posterior_samples, true_theta, filename):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Estimate the density for the prior samples
    x_prior = prior_samples[:, 0]
    y_prior = prior_samples[:, 1]
    kde_prior = gaussian_kde(np.vstack([x_prior, y_prior]))
    density_prior = kde_prior(np.vstack([x_prior, y_prior]))

    # Create a grid of x and y values
    x_grid_prior, y_grid_prior = np.meshgrid(np.linspace(x_prior.min(), x_prior.max(), num=100),
                                             np.linspace(y_prior.min(), y_prior.max(), num=100))
    z_grid_prior = kde_prior(np.vstack([x_grid_prior.ravel(), y_grid_prior.ravel()]))
    density_prior = z_grid_prior.reshape(x_grid_prior.shape)

    # Generate the contour plot for the prior samples in the first subplot
    levels_prior = np.linspace(density_prior.min(), density_prior.max(), num=10)
    ax1.contour(x_grid_prior, y_grid_prior, density_prior, levels=levels_prior, cmap='viridis')
    ax1.set_title("Prior Contour Density")
    ax1.set_xlabel("\u03B8\u2081")
    ax1.set_ylabel("\u03B8\u2080")

    # Estimate the density for the posterior samples
    x_posterior = posterior_samples[:, 0]
    y_posterior = posterior_samples[:, 1]
    kde_posterior = gaussian_kde(np.vstack([x_posterior, y_posterior]))
    density_posterior = kde_posterior(np.vstack([x_posterior, y_posterior]))

    # Create a grid of x and y values
    x_grid_posterior, y_grid_posterior = np.meshgrid(np.linspace(x_posterior.min(), x_posterior.max(), num=100),
                                                     np.linspace(y_posterior.min(), y_posterior.max(), num=100))
    z_grid_posterior = kde_posterior(np.vstack([x_grid_posterior.ravel(), y_grid_posterior.ravel()]))
    density_posterior = z_grid_posterior.reshape(x_grid_posterior.shape)

    # Generate the contour plot for the posterior samples in the second subplot
    levels_posterior = np.linspace(density_posterior.min(), density_posterior.max(), num=10)
    ax2.contour(x_grid_posterior, y_grid_posterior, density_posterior, levels=levels_posterior, cmap='viridis')
    ax2.set_title("Posterior Contour Density")
    ax2.set_xlabel("\u03B8\u2081")
    ax2.set_ylabel("\u03B8\u2080")


def plot_prior_posteriors(prior_samples, posterior_samples, posterior_samples1, posterior_samples2, true_theta, filename,
                         bandwidth=0.5):
    # Create a figure with four subplots in a row
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Estimate the density for the prior samples
    x_prior = prior_samples[:, 0]
    y_prior = prior_samples[:, 1]
    kde_prior = gaussian_kde(np.vstack([x_prior, y_prior]), bw_method=bandwidth)
    density_prior = kde_prior(np.vstack([x_prior, y_prior]))

    # Create a grid of x and y values for the prior samples
    x_grid_prior, y_grid_prior = np.meshgrid(np.linspace(x_prior.min(), x_prior.max(), num=100),
                                             np.linspace(y_prior.min(), y_prior.max(), num=100))
    z_grid_prior = kde_prior(np.vstack([x_grid_prior.ravel(), y_grid_prior.ravel()]))
    density_prior = z_grid_prior.reshape(x_grid_prior.shape)

    # Generate the contour plot for the prior samples
    levels_prior = np.linspace(density_prior.min(), density_prior.max(), num=10)
    axes[0].contour(x_grid_prior, y_grid_prior, density_prior, levels=levels_prior, cmap='viridis')
    axes[0].set_title("Prior Density")
    axes[0].set_xlabel("\u03B8\u2081")
    axes[0].set_ylabel("\u03B8\u2080")
    axes[0].set_xlim(-10, 10)
    axes[0].set_ylim(-10, 10)
    axes[0].plot(5, 2, marker='x', color='red', markersize=10)

    # Define the posterior samples and their labels
    posteriors = [
        {"samples": posterior_samples, "label": "D=1"},
        {"samples": posterior_samples1, "label": "D=10"},
        {"samples": posterior_samples2, "label": "D=100"}
    ]

    # Iterate over the posteriors and generate the contour plots
    for i, posterior in enumerate(posteriors, start=1):
        samples = posterior["samples"]
        x_posterior = samples[:, 0]
        y_posterior = samples[:, 1]
        kde_posterior = gaussian_kde(np.vstack([x_posterior, y_posterior]), bw_method=bandwidth)
        density_posterior = kde_posterior(np.vstack([x_posterior, y_posterior]))

        # Create a grid of x and y values for the posterior samples
        x_grid_posterior, y_grid_posterior = np.meshgrid(np.linspace(x_posterior.min(), x_posterior.max(), num=100),
                                                         np.linspace(y_posterior.min(), y_posterior.max(), num=100))
        z_grid_posterior = kde_posterior(np.vstack([x_grid_posterior.ravel(), y_grid_posterior.ravel()]))
        density_posterior = z_grid_posterior.reshape(x_grid_posterior.shape)

        # Generate the contour plot for the posterior samples
        levels_posterior = np.linspace(density_posterior.min(), density_posterior.max(), num=10)
        ax = axes[i]
        ax.contour(x_grid_posterior, y_grid_posterior, density_posterior, levels=levels_posterior, cmap='viridis')
        ax.set_title(f"{posterior['label']} Density")
        ax.set_xlabel("\u03B8\u2081")
        # ax.set_ylabel("\u03B8\u2080")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.plot(5, 2, marker='x', color='red', markersize=10)

    # Adjust the spacing between subplots
    # plt.subplots_adjust(wspace=0.4)

    # Save the plot as a PNG file with the provided filename
    plt.savefig(filename, dpi=900, bbox_inches='tight')



def plot_prior_posterior(prior_samples, posterior_samples, true_theta, filename):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first histogram in the first subplot
    ax1.hist(prior_samples[:, 0], bins=30)
    ax1.hist(posterior_samples[:, 0], bins=30, alpha=0.5, color='orange', label='Posterior')
    ax1.axvline(posterior_samples[:, 0].mean(), color='g', linestyle='--')
    ax1.axvline(true_theta[0][0], color='r', linestyle='--')
    ax1.set_title("\u03B8\u2081")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")

    # Plot the second histogram in the second subplot
    ax2.hist(prior_samples[:, 1], bins=30)
    ax2.hist(posterior_samples[:, 1], bins=30, alpha=0.5, color='orange', label='Posterior')
    ax2.axvline(posterior_samples[:, 1].mean(), color='g', linestyle='--')
    ax2.axvline(true_theta[0][1], color='r', linestyle='--')
    ax2.set_title("\u03B8\u2080")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")

    # Save the plot as a PNG file with the provided filename
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Display the figure
    # plt.show()


def save_posterior_marginal(posterior_samples_marginal, filename):
    # Create a figure
    fig = plt.figure(figsize=(8, 6))
    
    # Plot the histogram
    plt.hist(posterior_samples_marginal, bins=50)
    
    # Set labels and title
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Posterior Marginal")
    
    # Save the plot as a PNG file with the provided filename
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    
@jax.jit
def inverse_standard_scale(scaled_x, shift, scale):
    return (scaled_x * scale) + shift


@jax.jit
def standard_scale(x):
    def single_column_fn(x):
        mean = jnp.mean(x)
        std = jnp.std(x) + 1e-10
        return (x - mean) / std
        
    def multi_column_fn(x):
        mean = jnp.mean(x, axis=0, keepdims=True)
        std = jnp.std(x, axis=0, keepdims=True) + 1e-10
        return (x - mean) / std
        
    scaled_x = jax.lax.cond(
        x.shape[-1] == 1,
        single_column_fn,
        multi_column_fn,
        x
    )
    return scaled_x


def jax_lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    if jnp.isscalar(A):
        A = A * jnp.ones(dimensions)
        return A
    shape = tuple(dimensions) + A.shape
    A = A[jnp.newaxis, ...]
    A = jnp.broadcast_to(A, shape)
    return A


def sir_update(log_likelihood_fn, prior_samples, prior_log_probs, prng_key, 
               likelihood_params, x_obs, xi):
    log_likelihoods = log_likelihood_fn.apply(likelihood_params, x_obs, prior_samples, xi)
    
    # Update the importance weights
    new_log_weights = prior_log_probs + log_likelihoods
    
    # Normalize the weights
    max_log_weight = jnp.max(new_log_weights)
    log_weights_shifted = new_log_weights - max_log_weight
    unnormalized_weights = jnp.exp(log_weights_shifted)
    
    # Resample with the updated weights
    posterior_weights = unnormalized_weights / jnp.sum(unnormalized_weights)
    posterior_samples = jrandom.choice(prng_key, prior_samples, shape=(len(prior_samples),), replace=True, p=posterior_weights)
    
    return posterior_samples, posterior_weights


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


