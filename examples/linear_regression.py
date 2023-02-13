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
from functools import partial
import optax
import distrax
import haiku as hk

import tensorflow as tf
import tensorflow_datasets as tfds

from lfiax.flows.nsf import make_nsf
from lfiax.utils.oed_losses import lfi_pce_eig_scan
# from lfiax.utils.utils import jax_lexpand

from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Callable,
)

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


# @jax.jit
# def update(
#     params: hk.Params, prng_key: PRNGKey, opt_state: OptState, batch: Batch
# ) -> Tuple[hk.Params, OptState]:
#     """Single SGD update step."""
#     x, theta, d, xi = prepare_tf_dataset(batch)
#     # Note that `xi` is passed as a parameter to be updated during optimization
#     grads = jax.grad(unified_loss_fn)(params, prng_key, x, theta)
#     updates, new_opt_state = optimizer.update(grads, opt_state)
#     new_params = optax.apply_updates(params, updates)
#     return new_params, new_opt_state


def main():
    seed = 1231
    M = 3
    key = jrandom.PRNGKey(seed)

    # d = jnp.array([1.])
    d = jnp.array([])
    xi = jnp.array([0.])
    d_sim = jnp.concatenate((d, xi), axis=0)

    # Actually necessary value for `sim_data` output
    len_xi = len(xi)

    # Params and hyperparams
    theta_shape = (2,)
    # d_shape = (len(d),)
    xi_shape = (len_xi,)
    EVENT_SHAPE = (len(d_sim),)
    # EVENT_DIM is important for the normalizing flow's block.
    EVENT_DIM = 1

    num_samples = 10
    flow_num_layers = 5 #3 # 10
    mlp_num_layers = 1 # 3 # 4
    hidden_size = 128 # 500
    num_bins = 4
    learning_rate = 1e-4

    training_steps = 1000
    # eval_frequency = 5

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


    def lagrange_weight_schedule(iteration, decay_rate=0.99):
        """
        Returns the Lagrange multiplier weight for a given iteration using an exponential decay.
        """
        return decay_rate ** iteration


    @partial(jax.jit, static_argnums=[3,4])
    def update_pce(
        params: hk.Params, prng_key: PRNGKey, opt_state: OptState, N: int, M: int, designs: Array,
    ) -> Tuple[hk.Params, OptState]:
        """Single SGD update step."""
        xi_broadcast = jnp.broadcast_to(params["xi"], (N, len(xi)))
        log_prob_fun = lambda x, theta: log_prob.apply(
            params, x, theta, xi_broadcast)
        # Bingo. This is where to change data generation process
        loss, grads = jax.value_and_grad(lfi_pce_eig_scan)(
            params, prng_key, log_prob_fun, designs, N=N, M=M)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss


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

    # loss_deque = deque(maxlen=early_stopping_memory)
    for step in range(training_steps):
        params, opt_state, loss = update_pce(
            params, next(prng_seq), opt_state, N=num_samples, M=M, designs=d_sim
        )
        # Update d_sim vector
        d_sim = jnp.concatenate((d, params['xi']), axis=0)

        print(f"STEP: {step:5d}; Xi: {params['xi']}; Loss: {loss}")


# class Workspace:
#     def __init__(self, cfg):
#     def run(self) -> Callable:
#     def save(self, tag='latest'):
#     def _init_logging(self):

# from linear_regression import Workspace as W

# @hydra.main(config_name="config")
# def main(cfg)


if __name__ == "__main__":
    main()
