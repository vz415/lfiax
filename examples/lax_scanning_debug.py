# from omegaconf import DictConfig, OmegaConf
# import hydra
from collections import deque
import math
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
# from jax.test_util import check_grads

import numpy as np
from functools import partial
import optax
import distrax
import haiku as hk

import tensorflow as tf
import tensorflow_datasets as tfds

from lfiax.flows.nsf import make_nsf

from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


def jax_lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    if jnp.isscalar(A):
        A = A * jnp.ones(dimensions)
        return A
    shape = tuple(dimensions) + A.shape
    A = A[jnp.newaxis, ...]
    A = jnp.broadcast_to(A, shape)
    return A

# ----------------------------------------
# Prior simulators
# ----------------------------------------
def sim_linear_prior(num_samples: int, key: PRNGKey):
    """
    Simulate prior samples and return their log_prob.
    """
    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    base_distribution = distrax.Independent(
        distrax.MultivariateNormalDiag(mu, sigma)
    )

    samples, log_prob = base_distribution.sample_and_log_prob(seed=key, sample_shape=[num_samples])

    return samples, log_prob

# ----------------------------------------
# Likelihood simulators
# ----------------------------------------
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

    # perform forward pass
    y = jnp.broadcast_to(priors[:, 0], (len(d), len(priors)))
    y = y + jnp.expand_dims(d, 1) @ jnp.expand_dims(priors[:, 1], 0)
    y = y + n_g + jnp.squeeze(n_n)
    ygrads = priors[:, 1]

    return y, ygrads


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

    n_n = distrax.Gamma(concentration, rate).sample(seed=keys[0], sample_shape=[len(d), len(priors)])

    # perform forward pass
    y = jnp.broadcast_to(priors[:, 0], (len(d), len(priors)))
    y = distrax.MultivariateNormalDiag(y, jnp.squeeze(n_n)).sample(seed=keys[1], sample_shape=())

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


def sim_data(d: Array, num_samples: Array, key: PRNGKey):
    """
    Returns data in a format suitable for normalizing flow training.
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


# ----------------------------------------
# Helper functions to simulate data
# ----------------------------------------
def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
    ds = split
    ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1000)
    ds = ds.repeat()
    return iter(tfds.as_numpy(ds))


def prepare_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
    # Batch is [y, thetas, d]
    data = batch.astype(np.float32)
    x = data[:, :len_x]
    cond_data = data[:, len_x:]
    theta = cond_data[:, :-len_x]
    d = cond_data[:, -len_x:-len_xi]
    xi = cond_data[:, -len_xi:]
    return x, theta, d, xi


# ----------------------------
# Haiku transform functions for training and evaluation
# ----------------------------
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


@hk.without_apply_rng
@hk.transform
def model_sample(key: PRNGKey, num_samples: int, theta: Array, xi: Array) -> Array:
    model = make_nsf(
        event_shape=EVENT_SHAPE,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_x=False,
        standardize_theta=False,
        use_resnet=True,
        event_dim=EVENT_DIM,
    )
    return model._sample_n(key=key, n=[num_samples], theta=theta, xi=xi)


def loss_fn(
    params: hk.Params, prng_key: PRNGKey, x: Array, theta: Array, d: Array, xi: Array
) -> Array:
    loss = -jnp.mean(log_prob.apply(params, x, theta, d, xi))
    return loss


def unified_loss_fn(
    params: hk.Params, prng_key: PRNGKey, x: Array, theta: Array
) -> Array:
    xi = jnp.asarray(params['xi'])
    xi = jnp.broadcast_to(xi, (len(x), len(xi)))
    flow_params = {k: v for k, v in params.items() if k != 'xi'}
    
    # Loss is average negative log likelihood.
    loss = -jnp.mean(log_prob.apply(flow_params, x, theta, xi))
    return loss


@jax.jit
def eval_fn(params: hk.Params, batch: Batch) -> Array:
    x, theta, d, xi = prepare_data(batch)
    loss = -jnp.mean(log_prob.apply(params, x, theta, xi))
    return loss


@jax.jit
def update(
    params: hk.Params, prng_key: PRNGKey, opt_state: OptState, batch: Batch
) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    x, theta, d, xi = prepare_data(batch)
    # Note that `xi` is passed as a parameter to be updated during optimization
    grads = jax.grad(unified_loss_fn)(params, prng_key, x, theta)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def lfi_pce_eig(params: hk.Params, prng_key: PRNGKey, N: int=100, M: int=10, **kwargs):
    keys = jrandom.split(prng_key, 3 + M)
    xi = params['xi']
    flow_params = {k: v for k, v in params.items() if k != 'xi'}

    # simulate the outcomes before finding their log_probs
    X = sim_data(d_sim, num_samples, keys[0])  # Do I need to split up the prng_key?

    # I'm implicitly returning the prior here, that's a little annoying...
    x, theta_0, d, xi = prepare_data(X)  # TODO: Maybe refactor this?

    conditional_lp = log_prob.apply(flow_params, x, theta_0, d, xi)

    contrastive_lps = []
    thetas = []
    # TODO: Make this jax.lax expression for safe tracing and execution
    for i in range(M):
        # breakpoint()
        theta, _ = sim_linear_prior(num_samples, keys[i + 1])
        thetas.append(theta)
        contrastive_lp = log_prob.apply(flow_params, x, theta, d, xi)
        contrastive_lps.append(contrastive_lp)

    marginal_log_prbs = jnp.concatenate((jax_lexpand(conditional_lp, 1), jnp.array(contrastive_lps)))

    marginal_lp = jax.nn.logsumexp(marginal_log_prbs, 0) - math.log(M + 1)

    return - sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)


@jax.jit
def update_pce(
    params: hk.Params, prng_key: PRNGKey, opt_state: OptState, N: int, M: int
) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(lfi_pce_eig)(params, prng_key, N=num_samples, M=inner_samples)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


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

    base_distribution = distrax.Independent(  # Should this be independent?
        distrax.MultivariateNormalDiag(mu, sigma)
    )

    priors = base_distribution.sample(seed=keys[0], sample_shape=[num_samples])

    # Simulating noise and response
    noise_shape = (1,)

    mu_noise = jnp.zeros(noise_shape)
    sigma_noise = jnp.ones(noise_shape)

    n_n = distrax.Independent(
        distrax.MultivariateNormalDiag(mu_noise, sigma_noise)
    ).sample(seed=keys[1], sample_shape=[len(d), len(priors)])

    # sample random gamma noise
    n_g = distrax.Gamma(2.0, 0.5).sample(
        seed=keys[2], sample_shape=[len(d), len(priors)]
    )

    # perform forward pass
    y = jax.vmap(partial(jnp.dot, priors[:, 0]))(d)
    y = jax.vmap(partial(jnp.add, priors[:, 1]))(y) + n_g + jnp.squeeze(n_n)
    ygrads = priors[:, 1]

    return y.T, jnp.squeeze(priors)


@partial(jax.jit, static_argnums=0)
def sim_linear_prior(num_samples: int, key: PRNGKey):
    """
    Simulate prior samples and return their log_prob.
    """
    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    base_distribution = distrax.Independent(
        distrax.MultivariateNormalDiag(mu, sigma)
    )

    samples, log_prob = base_distribution.sample_and_log_prob(seed=key, sample_shape=[num_samples])

    return samples, log_prob


def loop_body_fun(carry, i):
    thetas, contrastive_lps, num_samples, keys, log_prob, flow_params, x, xi_broadcast = carry
    theta, _ = sim_linear_prior(num_samples, keys[i + 1])
    thetas = jnp.concatenate((thetas, jnp.expand_dims(theta, 0)))
    contrastive_lp = log_prob.apply(flow_params, x, theta, xi_broadcast)
    contrastive_lps = jnp.concatenate((contrastive_lps, jnp.expand_dims(contrastive_lp, 0)))
    return (thetas, contrastive_lps, num_samples, keys, log_prob, flow_params, x, xi_broadcast)


def compute_marginal_lp(M, num_samples, keys, log_prob, flow_params, x, xi_broadcast, conditional_lp):
    init_carry = (jnp.array([]), jnp.array([]), num_samples, keys, log_prob, flow_params, x, xi_broadcast)
    _, (thetas, contrastive_lps, _, _, _, _, _, _) = lax.fori_loop(0, M, loop_body_fun, init_carry)

    marginal_log_prbs = jnp.concatenate((jnp.expand_dims(conditional_lp, 1), contrastive_lps))
    marginal_lp = jnp.logsumexp(marginal_log_prbs, 0) - jnp.log(M + 1)

    return marginal_lp


def compute_marginal_lp2(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp):
    def loop_body_fun2(i, carry):
        # contrastive_lps, num_samples, keys, flow_params, x, xi_broadcast = carry
        contrastive_lps = carry
        theta, _ = sim_linear_prior(num_samples, keys[i + 1])
        contrastive_lp = log_prob.apply(flow_params, x, theta, xi_broadcast)
        print(contrastive_lp.shape)
        contrastive_lps += jnp.expand_dims(jnp.exp(contrastive_lp), 0)
        return (contrastive_lps, num_samples, keys, flow_params, x, xi_broadcast)
    # conditional_lps = jax.lax.fori_loop(0, M, loop_body_fun2, (conditional_lp, num_samples, keys, flow_params, x, xi_broadcast))
    print(conditional_lp.shape)
    conditional_lps = jax.lax.fori_loop(0, M, loop_body_fun2, conditional_lp)
    return jnp.log(conditional_lps)


# @partial(jax.jit, static_argnums=[2,3])
def lfi_pce_eig(params: hk.Params, prng_key: PRNGKey, N: int=100, M: int=10, **kwargs):
    keys = jrandom.split(prng_key, 3 + M)
    xi = params['xi']
    flow_params = {k: v for k, v in params.items() if k != 'xi'}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])
    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    contrastive_lps = []
    thetas = []
    # TODO: Make this jax.lax expression for safe tracing and execution
    for i in range(M):
        # breakpoint()
        theta, _ = sim_linear_prior(num_samples, keys[i + 1])
        thetas.append(theta)
        contrastive_lp = log_prob.apply(flow_params, x, theta, xi_broadcast)
        contrastive_lps.append(contrastive_lp)

    marginal_log_prbs = jnp.concatenate((jax_lexpand(conditional_lp, 1), jnp.array(contrastive_lps)))

    marginal_lp = jax.nn.logsumexp(marginal_log_prbs, 0) - math.log(M + 1)

    return - sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)


def lfi_pce_eig_lax(params: hk.Params, prng_key: PRNGKey, N: int=100, M: int=10, **kwargs):
    keys = jrandom.split(prng_key, 3 + M)
    xi = params['xi']
    flow_params = {k: v for k, v in params.items() if k != 'xi'}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])
    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    # conditional_lp could be the initial starting state that is added upon... 
    marginal_lp = compute_marginal_lp2(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp)

    return - sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)


@jax.jit
def update_pce(
    params: hk.Params, prng_key: PRNGKey, opt_state: OptState, N: int, M: int
) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(lfi_pce_eig)(params, prng_key, N=num_samples, M=inner_samples)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


if __name__ == "__main__":
    seed = 1231
    key = jrandom.PRNGKey(seed)
    M = 10
    keys = jrandom.split(key, 3 + M)
    num_samples = 10

    d = jnp.array([1.])
    xi = jnp.array([0.])
    d_sim = jnp.concatenate((d, xi), axis=0)

    # helper variables for `prepare_tf_dataset`
    len_x = len(d_sim)
    # # Actually necessary value for `sim_data` output
    len_xi = len(xi)

    # Params and hyperparams
    theta_shape = (2,)
    xi_shape = (len_xi,)
    EVENT_SHAPE = (len(d_sim),)
    # EVENT_DIM is important for the normalizing flow's block.
    EVENT_DIM = 1

    num_samples = 2
    inner_samples = 10 # AKA M or L in BOED parlance
    batch_size = 128
    flow_num_layers = 5 #3 # 10
    mlp_num_layers = 1 # 3 # 4
    hidden_size = 128 # 500
    num_bins = 4
    learning_rate = 1e-4
    warmup_steps = 10
    early_stopping_memory = 10
    early_stopping_threshold = 5e-2

    training_steps = 100
    eval_frequency = 5

    prng_seq = hk.PRNGSequence(42)  # TODO: Put one of "keys" here?
    params = log_prob.init(
        next(prng_seq),
        np.zeros((1, *EVENT_SHAPE)),
        np.zeros((1, *theta_shape)),
        np.zeros((1, *xi_shape)),
    )
    params['xi'] = xi

    optimizer = optax.adam(learning_rate)

    lfi_pce_eig_lax(params, key, N=3, M=M)
