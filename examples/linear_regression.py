# from omegaconf import DictConfig, OmegaConf
# import hydra
from collections import deque
import math

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
# from jax.test_util import check_grads

import numpy as np
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


# class Workspace:
#     def __init__(self, cfg):
#     def run(self) -> Callable:
#     def save(self, tag='latest'):
#     def _init_logging(self):

# from linear_regression import Workspace as W

# @hydra.main(config_name="config")
# def main(cfg)


if __name__ == "__main__":
    # main()
    # TODO: Put this in hydra config file
    seed = 1231
    M = 3
    key = jrandom.PRNGKey(seed)

    d = jnp.array([])
    xi = jnp.array([0.])
    d_sim = jnp.concatenate((d, xi), axis=0)

    # Params and hyperparams
    len_x = len(d_sim)
    len_d = len(d)
    len_xi = len(xi)

    theta_shape = (2,)
    d_shape = (len(d),)
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

    # Initialize the params
    prng_seq = hk.PRNGSequence(42)  # TODO: Put one of "keys" here?
    params = log_prob.init(
        next(prng_seq),
        np.zeros((1, *EVENT_SHAPE)),
        np.zeros((1, *theta_shape)),
        np.zeros((1, *xi_shape)),
    )
    params['xi'] = xi

    optimizer = optax.adam(learning_rate)

    opt_state = optimizer.init(params)

    # TODO: put this function in training since d will be changing.
    X_samples = 512*20
    X = sim_data(d_sim, X_samples, key)

    shift = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-14

    # Create tf dataset from sklearn dataset
    dataset = tf.data.Dataset.from_tensor_slices(X)

    # Splitting into train/validate ds
    train = dataset.skip(2000)
    val = dataset.take(2000)

    # load_dataset(split: tfds.Split, batch_size: int)
    train_ds = load_dataset(train, 512)
    valid_ds = load_dataset(val, 512)

    loss_deque = deque(maxlen=early_stopping_memory)
    for step in range(training_steps):
        # params, opt_state = update_pce(
        #     params, next(prng_seq), opt_state, N=num_samples, M=M
        # )
        params, opt_state = update(
            params, next(prng_seq), opt_state, next(train_ds)
        )

        print(f"STEP: {step:5d}; Xi: {params['xi']}")
        if step % eval_frequency == 0:
            val_loss = eval_fn(params, next(valid_ds))
            print(f"STEP: {step:5d}; Validation loss: {val_loss:.3f}")
        
            loss_deque.append(val_loss)
            avg_abs_diff = jnp.mean(abs(jnp.array(loss_deque) - sum(loss_deque)/len(loss_deque)))
            if step > warmup_steps and avg_abs_diff < early_stopping_threshold:
                break
