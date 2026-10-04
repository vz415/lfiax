import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom

from functools import partial
import distrax
import haiku as hk

from lfiax.utils.utils import get_calibration_error_jax

from typing import Mapping, Any, Tuple, Callable

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


def kl_loss_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch,
               x_train_mean: Array, x_train_std: Array, theta_train_mean: Array,
               theta_train_std: Array, log_prob: Callable) -> Array:
    """Really only for the 2 moons task or one where you normalize by dataset stats."""
    x = batch['x']
    theta = batch['theta']
    x = (x - x_train_mean) / x_train_std
    theta = (theta - theta_train_mean) / theta_train_std
    dummy_xi = jnp.zeros((x.shape[0], 0))
    nll = -jnp.mean(log_prob.apply(params, x, theta, dummy_xi))
    loss = nll
    return loss


@partial(jax.jit, static_argnums=[4])
def kl_loss_fn_general(params: hk.Params, x: Array, theta: Array, xi: Array,
                       log_prob: Callable) -> Array:
    """Expects the xi to be curried with the logprob function."""
    return -jnp.mean(log_prob(params, x, theta, xi))


@partial(jax.jit, static_argnums=[5])
def kl_loss_fn_dropout(params: hk.Params, prng_key: PRNGKey, x: Array, theta: Array, xi: Array,
                       log_prob: Callable) -> Array:
    """Expects the xi to be curried with the logprob function."""
    # return -jnp.mean(log_prob(params, x, theta, xi))
    return -jnp.mean(log_prob(params, prng_key, x, theta, xi))


def kl_sbc_loss_2moons_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch,
                   x_train_mean: Array, x_train_std: Array, theta_train_mean: Array,
                   theta_train_std: Array, log_prob: Callable, sbc_n_samples: int, sbc_lambda: float
                   ) -> Array:
    x = batch['x']
    theta = batch['theta']
    x = (x - x_train_mean) / x_train_std
    theta = (theta - theta_train_mean) / theta_train_std
    dummy_xi = jnp.zeros((x.shape[0], 0))
    nll = -jnp.mean(log_prob.apply(params, x, theta, dummy_xi))

    # Define model function with current params
    def model_fn(theta_input, x_input):
        # Ensure inputs are standardized
        x_input = (x_input - x_train_mean) / x_train_std
        theta_input = (theta_input - theta_train_mean) / theta_train_std
        dummy_xi_local = jnp.zeros((theta_input.shape[0], 0))
        return log_prob.apply(params, x_input, theta_input, dummy_xi_local)

    # Define prior sampling function for TWO MOONS only
    def prior_sample(prng_key, n_samples):
        # Assuming prior is Uniform over [-1, 1] x [-1, 1] for the "two_moons" task
        lower = jnp.array([-1.0, -1.0])
        upper = jnp.array([1.0, 1.0])
        prior_samples = jrandom.uniform(prng_key, (n_samples, 2), minval=lower, maxval=upper)
        # Standardize prior samples
        prior_samples = (prior_samples - theta_train_mean) / theta_train_std
        # Compute log probabilities (uniform distribution)
        prior_lps = -jnp.sum(jnp.log(upper - lower))
        prior_lps = jnp.full((n_samples,), prior_lps)
        return prior_samples, prior_lps

    # Compute calibration error
    sbc_n_samples = sbc_n_samples
    sbc_lambda = sbc_lambda
    prior_samples, prior_lps = prior_sample(prng_key, sbc_n_samples)
    calibration_error = get_calibration_error_jax(
        model_fn, theta, x, prior_samples, prior_lps, calibration=1)

    # Total loss
    loss = nll + sbc_lambda * calibration_error

    return loss

def kl_sbc_loss_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch, prior_sample: Callable,
                   x_train_mean: Array, x_train_std: Array, theta_train_mean: Array,
                   theta_train_std: Array, log_prob: Callable, sbc_n_samples: int, sbc_lambda: float
                   ) -> Array:
    x = batch['x']
    theta = batch['theta']
    x = (x - x_train_mean) / x_train_std
    theta = (theta - theta_train_mean) / theta_train_std
    dummy_xi = jnp.zeros((x.shape[0], 0))
    nll = -jnp.mean(log_prob.apply(params, x, theta, dummy_xi))

    # Define model function with current params
    def model_fn(theta_input, x_input):
        # Ensure inputs are standardized
        x_input = (x_input - x_train_mean) / x_train_std
        theta_input = (theta_input - theta_train_mean) / theta_train_std
        dummy_xi_local = jnp.zeros((theta_input.shape[0], 0))
        # print("x_input shape: ", x_input.shape)
        # print("theta_input shape: ", theta_input.shape)
        return log_prob.apply(params, x_input, theta_input, dummy_xi_local)

    # Compute calibration error
    sbc_n_samples = sbc_n_samples
    sbc_lambda = sbc_lambda
    # currying the prng_key or just doing a numpy function
    prior_samples, prior_lps = prior_sample(sbc_n_samples)
    calibration_error = get_calibration_error_jax(
        model_fn, theta, x, prior_samples, prior_lps, calibration=1)

    # Total loss
    loss = nll + sbc_lambda * calibration_error
    # loss = nll - sbc_lambda * calibration_error # TODO: see if this is the right one...

    return loss


def kl_sbc_loss_fn_general(params: hk.Params, x: Array, theta: Array, prior_sample: Callable,
                           log_prob: Callable, sbc_n_samples: int, sbc_lambda: float,
                           xi: Array = None
                           ) -> Array:
    """
    Assumes normalized inputs x, theta, and optionally xi.
    Computes the total loss as a combination of negative log-likelihood (NLL) and
    Simulation-Based Calibration (SBC) error, with optional `xi` variables.

    The function assumes that the inputs `x`, `theta`, and optionally `xi` are normalized
    beforehand. It first calculates the NLL of the provided log probability function with the
    given parameters and data. Then, it computes the SBC calibration error using prior samples
    and adjusts the total loss by a weighting factor `sbc_lambda`.
    """
    if xi is None:
        xi = jnp.zeros((x.shape[0], 0))

    nll = -jnp.mean(log_prob(params, x, theta, xi))

    # Define model function with current params
    def model_fn(theta_input, x_input):
        # depends on the shape of xi aka number of observations
        return log_prob(params, x_input, theta_input, xi[:x_input.shape[0]])

    # Compute calibration error
    sbc_n_samples = sbc_n_samples
    sbc_lambda = sbc_lambda
    # curry the prng_key or just doing a numpy function
    prior_samples, prior_lps = prior_sample(sbc_n_samples)
    calibration_error = get_calibration_error_jax(
        model_fn, theta, x, prior_samples, prior_lps, calibration=1)

    # Total loss
    loss = nll + sbc_lambda * calibration_error
    # loss = nll - sbc_lambda * calibration_error

    return loss


# Helper function for safe mean calculation
def _safe_mean_terms(terms):
    """Computes the mean of terms, avoiding NaNs."""
    finite_mask = jnp.isfinite(terms)
    num_finite = jnp.sum(finite_mask)
    safe_terms = jnp.where(finite_mask, terms, 0.0)
    mean = jnp.sum(safe_terms) / num_finite
    return mean, safe_terms


@partial(jax.jit, static_argnums=[3,4,5,6,11])
def lf_pce_eig_scan(flow_params: hk.Params,
                    prng_key: PRNGKey,
                    batch: Batch,
                    log_prob_fun: Callable,
                    N: int = 100,
                    M: int = 99,
                    lam: float = 0.1,
                    x_train_mean: Array = None,
                    x_train_std: Array = None,
                    theta_train_mean: Array = None,
                    theta_train_std: Array = None,
                    prior_sampler: Callable = None,
                    ):
    """
    Calculates LF-PCE loss using jax.lax.scan to accelerate.
    """
    keys = jrandom.split(prng_key, 2 + M)

    # TODO: maybe try alternative normalization
    # Normalize x and theta
    x = batch['x']
    theta_0 = batch['theta']
    scaled_x = (x - x_train_mean) / x_train_std
    theta_0 = (theta_0 - theta_train_mean) / theta_train_std
    # scaled_x = x

    # Create dummy xi array
    xi = jnp.zeros((x.shape[0], 0))

    def compute_marginal_lp(
            keys, log_prob_fun, M, theta_0, x, conditional_lp, prior_sampler):
        def scan_fun(carry, i):
            contrastive_lps, theta = carry
            if prior_sampler is None:
                theta = jnp.roll(theta, shift=1, axis=0)
            else:
                theta = prior_sampler(keys[i], theta.shape[0])
                if isinstance(theta, (tuple, list)):
                    theta = theta[0]
            contrastive_lp = log_prob_fun(flow_params, keys[i], x, theta, xi)
            contrastive_lps_next = jnp.logaddexp(contrastive_lps, contrastive_lp)
            return (contrastive_lps_next, theta), None

        initial_carry = (conditional_lp, theta_0)

        result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
        return result[0][0]

    conditional_lp = log_prob_fun(
        flow_params, keys[0], scaled_x, theta_0, xi
        )

    marginal_lp = compute_marginal_lp(
        keys[1:M+1], log_prob_fun, M, theta_0, scaled_x, conditional_lp, prior_sampler,
        )
    marginal_lp = marginal_lp - jnp.log(M + 1)

    # Compute EIG and loss
    eig_terms = conditional_lp - marginal_lp
    EIG, _ = _safe_mean_terms(eig_terms)
    loss = EIG + lam * jnp.mean(conditional_lp)

    return -loss, (conditional_lp, EIG)
