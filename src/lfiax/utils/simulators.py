import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom

import numpy as np
from functools import partial
import distrax
import haiku as hk

import torch

from lfiax.utils.utils import create_lognormal_to_gaussian_bijectors, sir_update_prod_likelihood_bespoke
from lfiax.utils.torch_utils import solve_sir_sdes

from functools import partial

from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Callable,
    NamedTuple
)

Array = jnp.ndarray
PRNGKey = Array


# -------- Linear regresssion model priors --------


@partial(jax.jit, static_argnums=0)
def sim_linear_prior(num_samples: int, key: PRNGKey):
    """
    Simulate prior samples and return their log_prob.
    """
    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    base_distribution = distrax.Independent(distrax.MultivariateNormalDiag(mu, sigma))

    samples, log_prob = base_distribution.sample_and_log_prob(
        seed=key, sample_shape=[num_samples]
    )

    return samples, log_prob

@partial(jax.jit, static_argnums=[0, 1])
def sim_linear_prior_M_samples(num_samples: int, M: int, key: PRNGKey):
    """
    Simulate prior samples and return their log_prob.
    """
    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    base_distribution = distrax.Independent(distrax.MultivariateNormalDiag(mu, sigma))

    samples, log_prob = base_distribution.sample_and_log_prob(
        seed=key, sample_shape=[M, num_samples]
    )

    return samples, log_prob

def sim_linear_jax(d: Array, priors: Array, key: PRNGKey):
    """
    Simulate linear model with normal and gamma noise, from Kleinegesse et al. 2020.
    """
    # Keys for the appropriate functions
    keys = jrandom.split(key, 3)

    # sample random normal dist
    noise_shape = (1,)

    mu_noise = jnp.zeros(noise_shape)
    sigma_noise = jnp.ones(noise_shape)

    n_n = distrax.Independent(
        distrax.MultivariateNormalDiag(mu_noise, sigma_noise)
    ).sample(seed=keys[0], sample_shape=[len(d), len(priors)])

    # sample random gamma noise
    n_g = distrax.Gamma(2.0, 1.0 / 2.0).sample(
        seed=keys[1], sample_shape=[len(d), len(priors)]
    )

    sigma = n_g + jnp.squeeze(n_n)

    # perform forward pass
    y = jnp.broadcast_to(priors[:, 0], (len(d), len(priors)))
    y = y + jnp.expand_dims(d, 1) @ jnp.expand_dims(priors[:, 1], 0)
    y = y + sigma
    ygrads = priors[:, 1]

    return y, ygrads, sigma

def sim_linear_jax_laplace(d: Array, priors: Array, key: PRNGKey):
    """
    Sim linear laplace prior regression model.

    Returns:
        y: scalar value, or, array of scalars.
    """
    # Keys for the appropriate functions
    keys = jrandom.split(key, 3)

    # sample random normal dist
    noise_shape = (1,)

    concentration = jnp.ones(noise_shape)
    rate = jnp.ones(noise_shape)

    n_n = distrax.Gamma(concentration, rate).sample(
        seed=keys[0], sample_shape=[len(d), len(priors)]
    )

    # perform forward pass
    y = jnp.broadcast_to(priors[:, 0], (len(d), len(priors)))
    y = distrax.MultivariateNormalDiag(y, jnp.squeeze(n_n)).sample(
        seed=keys[1], sample_shape=()
    )

    return y

def sim_data_laplace(d: Array, priors: Array, key: PRNGKey):
    """
    Returns data in a format suitable for normalizing flow training.
    Data will be in shape [y, thetas]. The `y` variable can vary in size.
    Uses `sim_linear_jax_laplace` function.
    """
    keys = jrandom.split(key, 2)
    theta_shape = (1,)

    loc = jnp.zeros(theta_shape)
    scale = jnp.ones(theta_shape)

    # Leaving in case this fixes future dimensionality issues
    # base_distribution = distrax.Independent(
    #     distrax.Laplace(loc, scale)
    # )
    base_distribution = distrax.Laplace(loc, scale)

    priors = base_distribution.sample(seed=keys[0], sample_shape=[num_samples])

    y = sim_linear_jax_laplace(d, priors, keys[1])

    return jnp.column_stack(
        (y.T, jnp.squeeze(priors), jnp.broadcast_to(d, (num_samples, len(d))))
    )

@partial(jax.jit, static_argnums=1)
def sim_linear_data_vmap(d: Array, num_samples: Array, key: PRNGKey):
    """
    Returns data in a format suitable for normalizing flow training.
    Data will be in shape (y, thetas, d). The `y` variable can vary in size.
    Has a fixed prior.
    """
    keys = jrandom.split(key, 3)

    # Simulating the priors
    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    base_distribution = distrax.Independent(distrax.MultivariateNormalDiag(mu, sigma))

    priors = base_distribution.sample(seed=keys[0], sample_shape=[num_samples])

    # Simulating noise and response
    noise_shape = (1,)

    mu_noise = jnp.zeros(noise_shape)
    sigma_noise = jnp.ones(noise_shape)

    n_n = distrax.Independent(
        distrax.MultivariateNormalDiag(mu_noise, sigma_noise)
    ).sample(seed=keys[1], sample_shape=[len(priors), d.shape[-1]])

    # sample random gamma noise
    n_g = distrax.Gamma(2.0, 0.5).sample(
        seed=keys[2], sample_shape=[len(priors), d.shape[-1]]
    )

    sigma = n_g + jnp.squeeze(n_n, -1)

    # perform forward pass
    if d.shape[-1] == 1:
        # "d" becomes (2, 1) shape whenever passing lists.
        y = jnp.matmul(jnp.expand_dims(priors[:, 0], -1), jnp.expand_dims(d, 0))
    else:
        # Designs are a length-2 arrays.
        y = jnp.matmul(
            jnp.expand_dims(priors[:, 0], -1), jnp.expand_dims(d, 0)
        ).squeeze()
    y = jnp.add(jnp.expand_dims(priors[:, 1], -1), y)
    y_noised = jnp.add(y, sigma)

    ygrads = priors[:, 1]

    return y_noised, priors, y, sigma

@jax.jit
def sim_linear_data_vmap_theta(d: Array, theta: Array, key: PRNGKey):
    """
    Returns data in a format suitable for normalizing flow training.
    Data will be in shape (y, thetas, d). The `y` variable can vary in size.

    Uses theta drawn from theta. Theta should be in the shape [num_samples, 2].
    """
    keys = jrandom.split(key, 2)

    # Simulating noise and response
    noise_shape = (1,)

    mu_noise = jnp.zeros(noise_shape)
    sigma_noise = jnp.ones(noise_shape)

    n_n = distrax.Independent(
        distrax.MultivariateNormalDiag(mu_noise, sigma_noise)
    ).sample(seed=keys[0], sample_shape=[len(theta), d.shape[-1]])

    # sample random gamma noise
    n_g = distrax.Gamma(2.0, 0.5).sample(
        seed=keys[1], sample_shape=[len(theta), d.shape[-1]]
    )

    sigma = n_g + jnp.squeeze(n_n, -1)

    # perform forward pass
    if d.shape[-1] == 1:
        # "d" becomes (2, 1) shape whenever passing lists.
        y = jnp.matmul(jnp.expand_dims(theta[:, 0], -1), jnp.expand_dims(d, 0))
        # BUG: I'm not sure why this sometimes works and sometimes does not.
        # y = jnp.matmul(jnp.expand_dims(theta[:,0], -1), d.T)
    else:
        # Designs are a length-2 arrays.
        y = jnp.matmul(
            jnp.expand_dims(theta[:, 0], -1), jnp.expand_dims(d, 0)
        ).squeeze()
    y = jnp.add(jnp.expand_dims(theta[:, 1], -1), y)
    y_noised = jnp.add(y, sigma)

    ygrads = theta[:, 1]

    return y_noised, y, sigma

def sim_data_tf(d: Array, num_samples: Array, key: PRNGKey):
    """
    Returns data in a format suitable for normalizing flow training using
    TF datasets.
    Data will be in shape [y, thetas]. The `y` variable can vary in size.
    """
    keys = jrandom.split(key, 2)

    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    base_distribution = distrax.Independent(  # Should this be independent?
        distrax.MultivariateNormalDiag(mu, sigma)
    )

    priors = base_distribution.sample(seed=keys[0], sample_shape=[num_samples])

    # ygrads allows to be compared to other implementations (Kleinegesse et)
    y, ygrads = sim_linear_jax(d, priors, keys[1])

    return jnp.column_stack(
        (y.T, jnp.squeeze(priors), jnp.broadcast_to(d, (num_samples, len(d))))
    )

def sim_data(d: Array, num_samples: Array, key: PRNGKey):
    """
    Returns data in a format suitable for normalizing flow training.
    Data will be in shape (y, thetas, d). The `y` variable can vary in size.
    """
    keys = jrandom.split(key, 2)

    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    base_distribution = distrax.Independent(  # Should this be independent?
        distrax.MultivariateNormalDiag(mu, sigma)
    )

    priors = base_distribution.sample(seed=keys[0], sample_shape=[num_samples])

    # ygrads allows to be compared to other implementations (Kleinegesse et)
    y, ygrads = sim_linear_jax(d, priors, keys[1])

    return y.T, jnp.squeeze(priors), jnp.broadcast_to(d, (num_samples, len(d)))

# SIR and EPIG experiment helpers.


def sample_lognormal_with_log_probs(seed, num_samples):
    theta_loc = jnp.log(jnp.array([0.5, 0.1]))
    theta_covmat = jnp.eye(2) * 0.5 ** 2
    mvn_dist = distrax.MultivariateNormalFullCovariance(loc=theta_loc, covariance_matrix=theta_covmat)
    normal_samples = mvn_dist.sample(seed=seed, sample_shape=(num_samples,))
    lognormal_samples = jnp.exp(normal_samples)
    log_probs = mvn_dist.log_prob(normal_samples)
    return lognormal_samples, log_probs + normal_samples.sum(axis=-1)

def lognormal_log_prob(theta):
    theta_loc = jnp.log(jnp.array([0.5, 0.1]))
    theta_covmat = jnp.eye(2) * 0.5 ** 2
    mvn_dist = distrax.MultivariateNormalFullCovariance(
        loc=theta_loc, covariance_matrix=theta_covmat)
    bijector = distrax.Block(distrax.Lambda(lambda x: jnp.log(x)), ndims=1)
    new_thetas, transform_log_prob = bijector.forward_and_log_det(theta)
    log_probs = mvn_dist.log_prob(new_thetas)
    return log_probs - transform_log_prob


@jax.jit
def linear_interpolate_trajectories(
    xi: Array, ts: Array, ys: Array
) -> Array:
    """Interpolate paired trajectory values at continuous design times.

    Each flattened design ``xi[i]`` is paired with trajectory column
    ``ys[:, i]``. Designs outside the simulated time grid use the nearest
    endpoint value.
    """
    xi_flat = jnp.clip(xi.reshape(-1), ts[0], ts[-1])
    left = jnp.searchsorted(ts, xi_flat, side="right") - 1
    left = jnp.clip(left, 0, ts.shape[0] - 2)
    right = left + 1
    columns = jnp.arange(xi_flat.shape[0])

    t_left = ts[left]
    t_right = ts[right]
    y_left = ys[left, columns]
    y_right = ys[right, columns]
    weight = (xi_flat - t_left) / (t_right - t_left)
    return ((1.0 - weight) * y_left + weight * y_right).reshape(-1, 1)

@jax.jit
def simulate_sir(xi: Array, ts: Array, ys: Array) -> Array:
    """Evaluate paired SIR trajectories at continuous design times.

    Small negative values from numerical SDE integration are clamped to the
    physically valid infected-state boundary before likelihood preprocessing.
    """
    y = jnp.maximum(linear_interpolate_trajectories(xi, ts, ys), 0.0)
    return y, jnp.mean(y), jnp.std(y)


def collect_sufficient_sde_samples_prior(
        N: int,
        prior_samples: Array,
        prior_log_probs: Array,
        device: str,
        prng_seq: PRNGKey,
        ) -> Tuple[Array, Array, Array]:
    '''
    Collect sufficient samples from SDE simulator to use for training.

    N: int, number of samples to collect
    prior_samples: (num_samples, 2) array of prior samples
    prior_log_prob: (num_samples, ) array of prior log probs
    flow_params: hk.Params, parameters for the normalizing flow
    x_obs_scale: (num_samples, ) array of x_obs_scale
    xi_sim: (num_samples, ) array of xi_sim
    device: str, device to run torch on
    log_prob_fun: Callable, log prob function to use for SIR update
    prng_seq: PRNGKey, random key for SIR update

    Returns:
    final_ys: (num_samples, ) array of final ys
    post_samples: (num_samples, 2) array of posterior samples
    post_log_probs: (num_samples, ) array of posterior log probs
    '''
    all_ys = jnp.zeros((100000, 0))
    all_prior_samples = jnp.zeros((0, 2))
    all_prior_log_probs = jnp.zeros((0,))

    params_device = torch.tensor(np.asarray(prior_samples)).to(device)
    params_log_probs_device = torch.tensor(np.asarray(prior_log_probs)).to(device)

    while all_ys.shape[1] < N:
        print("Generating initial SIR SDE training data...")
        sde_dict = solve_sir_sdes(
                    num_samples=N,
                    device=device,
                    grid=100000,
                    save=False,
                    savegrad=False,
                    params=params_device,
                    params_log_probs=params_log_probs_device,
                    seed=next(prng_seq)
                )

        current_ys = sde_dict['ys']  # Get the actual data from your dictionary
        current_ys = jnp.array(current_ys.numpy())
        all_ys = jnp.concatenate((all_ys, current_ys), axis=1)  # Concatenate along the second dimension

        current_prior_samples = jnp.array(sde_dict['prior_samples'].numpy())
        all_prior_samples = jnp.concatenate((all_prior_samples, current_prior_samples), axis=0)

        current_prior_log_probs = jnp.array(sde_dict['prior_log_probs'].numpy())
        all_prior_log_probs = jnp.concatenate((all_prior_log_probs, current_prior_log_probs), axis=0)

    # After the loop, 'all_ys' will contain successful samples
    final_ys = all_ys[:, :N]
    post_samples = all_prior_samples[:N]
    post_log_probs = all_prior_log_probs[:N]

    return final_ys, post_samples, post_log_probs
