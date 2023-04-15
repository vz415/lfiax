import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import omegaconf

import os
import csv, time
import pickle as pkl
import math
import random
import joblib

import jax
import numpy as np
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
from lfiax.utils.oed_losses import lf_pce_eig_scan
from lfiax.utils.simulators import sim_linear_data_vmap, sim_linear_data_vmap_theta
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

# Helper functions
# TODO: Make prior outside of the simulator so you can sample and pass it around
def make_lin_reg_prior():
    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    prior = distrax.Independent(
        distrax.MultivariateNormalDiag(mu, sigma)
    )
    return prior


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

@jax.jit
def inverse_standard_scale(scaled_x, shift, scale):
    return (scaled_x + shift) * scale


# Getting hyperparams from hydra config file.

with initialize(version_base=None, config_path="./"):
    cfg = compose(config_name="config")

# seed = cfg.seed

if cfg.designs.d is None:
    d = jnp.array([])
    xi = jnp.array([cfg.designs.xi])
    d_sim = xi # jnp.array([cfg.designs.xi])
else:
    d = jnp.array([cfg.designs.d])
    xi = jnp.array([cfg.designs.xi])
    d_sim = jnp.concatenate((d, xi), axis=1)

# Bunch of event shapes needed for various functions
len_xi = xi.shape[-1]
xi_shape = (len_xi,)
theta_shape = (2,)
EVENT_SHAPE = (d_sim.shape[-1],)
EVENT_DIM = cfg.param_shapes.event_dim

# contrastive sampling parameters
M = cfg.contrastive_sampling.M
N = cfg.contrastive_sampling.N

# likelihood flow's params
flow_num_layers = cfg.flow_params.num_layers
mlp_num_layers = cfg.flow_params.mlp_num_layers
hidden_size = cfg.flow_params.mlp_hidden_size
num_bins = cfg.flow_params.num_bins

# vi flow's parameters
vi_flow_num_layers = cfg.vi_flow_params.num_layers
vi_mlp_num_layers = cfg.vi_flow_params.mlp_num_layers
vi_hidden_size = cfg.vi_flow_params.mlp_hidden_size
vi_num_bins = cfg.vi_flow_params.num_bins
vi_samples = cfg.vi_flow_params.vi_samples

# Optimization parameters
learning_rate = cfg.optimization_params.learning_rate
xi_lr_init = cfg.optimization_params.xi_learning_rate
training_steps = cfg.optimization_params.training_steps
xi_optimizer = cfg.optimization_params.xi_optimizer
xi_scheduler = cfg.optimization_params.xi_scheduler
xi_lr_end = 1e-4


# TODO: reduce boilerplate code.
# @hk.transform_with_state
@hk.without_apply_rng
@hk.transform
def log_prob(x: Array, theta: Array, xi: Array) -> Array:
    x_scaled = standard_scale(x)
    # If this is the wrong shape, grads don't flow :(
    if x.shape[-1] == 1:
        x_scaled = x_scaled.squeeze(0)
    # TODO: Pass more nsf parameters from config.yaml
    model = make_nsf(
        event_shape=EVENT_SHAPE,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_theta=True,
        use_resnet=True,
        conditional=True
    )
    return model.log_prob(x_scaled, theta, xi)

@hk.without_apply_rng
@hk.transform
def vi_log_prob(theta: Array) -> Array:
    theta_scaled = standard_scale(theta)
    model = make_nsf(
        event_shape=theta_shape,
        num_layers=vi_flow_num_layers,
        hidden_sizes=[vi_hidden_size] * vi_mlp_num_layers,
        num_bins=vi_num_bins,
        use_resnet=True,
        conditional=False
    )
    return model.log_prob(theta_scaled)

log_prob = log_prob
vi_log_prob = vi_log_prob

@hk.without_apply_rng
@hk.transform
def likelihood_sample(key: PRNGKey, num_samples: int,
                shift: Array, scale: Array,
                x: Array, theta: Array, xi: Array) -> Array:
    # Does sampling the likelihood require x?
    """vi is sampling the posterior distributuion so doesn't need
    conditional information. Just uses distrax bijector layers.
    """
    model = make_nsf(
        event_shape=EVENT_SHAPE,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_theta=True,
        use_resnet=True,
        conditional=True
    )
    samples = model._sample_n(key=key, 
                            n=[num_samples]
                            )
    return inverse_standard_scale(samples, shift, scale)

@hk.without_apply_rng
@hk.transform
def vi_sample(key: PRNGKey, num_samples: int,
                shift: Array, scale: Array) -> Array:
    """vi is sampling the posterior distributuion so doesn't need
    conditional information. Just uses distrax bijector layers.
    """
    model = make_nsf(
        event_shape=theta_shape,
        num_layers=vi_flow_num_layers,
        hidden_sizes=[vi_hidden_size] * vi_mlp_num_layers,
        num_bins=vi_num_bins,
        use_resnet=True,
        conditional=False
    )
    samples = model._sample_n(key=key, 
                            n=[num_samples]
                            )
    return inverse_standard_scale(samples, shift, scale)


# Creating the function to parallelize with joblib
# Pasting all of the run parameters
def create_locals(d):
    locals_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recursively create local variables for its leaf nodes
            locals_dict.update(create_locals(value))
        else:
            # If the value is not a dictionary, create a local variable with the key as its name
            locals_dict[key] = value
    return locals_dict


def main():
    def train_single_model(seed):
        # Call the recursive function to create local variables for all leaf nodes of the dictionary
        locals_dict = create_locals(omegaconf.OmegaConf.to_container(cfg, resolve=True))
        
        locals().update(locals_dict)
        
        @partial(jax.jit, static_argnums=[5,6])
        def update_pce(
            flow_params: hk.Params, xi_params: hk.Params, prng_key: PRNGKey, \
            opt_state: OptState, opt_state_xi: OptState, N: int, M: int, \
            designs: Array,
        ) -> Tuple[hk.Params, OptState]:
            """Single SGD update step."""
            log_prob_fun = lambda params, x, theta, xi: log_prob.apply(
                params, x, theta, xi)
            
            (loss, (conditional_lp, theta_0, x, x_noiseless, noise, EIG)), grads = jax.value_and_grad(
                lf_pce_eig_scan, argnums=[0,1], has_aux=True)(
                flow_params, xi_params, prng_key, log_prob_fun, designs, N=N, M=M
                )
            
            updates, new_opt_state = optimizer.update(grads[0], opt_state)
            xi_updates, xi_new_opt_state = optimizer2.update(grads[1], opt_state_xi)

            new_params = optax.apply_updates(flow_params, updates)
            new_xi_params = optax.apply_updates(xi_params, xi_updates)
            
            return new_params, new_xi_params, new_opt_state, xi_new_opt_state, loss, grads[1], xi_updates, conditional_lp, theta_0, x, x_noiseless, noise, EIG

        # Initialize the net's params
        prng_seq = hk.PRNGSequence(seed)
        params = log_prob.init(
            next(prng_seq),
            np.zeros((1, *EVENT_SHAPE)),
            np.zeros((1, *theta_shape)),
            np.zeros((1, *xi_shape)),
        )

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        # TODO: Put this up in the initialization code
        if xi_scheduler == "None":
            schedule = xi_lr_init
        elif xi_scheduler == "Linear":
            schedule = optax.linear_schedule(xi_lr_init, xi_lr_end, transition_steps=training_steps)
        elif xi_scheduler == "Exponential":
            schedule = optax.exponential_decay(
                init_value=xi_lr_init,
                transition_steps=training_steps,
                decay_rate=(xi_lr_end / xi_lr_init) ** (1 / training_steps),
                staircase=False
            )
        else:
            raise AssertionError("Specified unsupported scheduler.")

        if xi_optimizer == "Adam":
            optimizer2 = optax.adam(learning_rate=schedule)
        elif xi_optimizer == "SGD":
            optimizer2 = optax.sgd(learning_rate=schedule)
        elif xi_optimizer == "Yogi":
            optimizer2 = optax.yogi(learning_rate=schedule)
        elif xi_optimizer == "AdaBelief":
            optimizer2 = optax.adabelief(learning_rate=schedule)

        # This could be initialized by a distribution of designs!
        # params['xi'] = xi
        # BUG: For some reason locals() would not update the local `xi` parameter, so using it directly.
        params['xi'] = locals_dict['xi']
        xi_params = {key: value for key, value in params.items() if key == 'xi'}

        # Normalize xi values for optimizer
        design_min = -10.
        design_max = 10.
        scale_factor = float(jnp.max(jnp.array([jnp.abs(design_min), jnp.abs(design_max)])))
        xi_params_max_norm = {}
        xi_params_max_norm['xi'] = jnp.divide(xi_params['xi'], scale_factor)
        # xi_params_scaled = (xi_params['xi'] - jnp.mean(xi_params['xi'])) / (jnp.std(xi_params['xi']) + 1e-10)

        opt_state_xi = optimizer2.init(xi_params_max_norm)
        flow_params = {key: value for key, value in params.items() if key != 'xi'}

        for step in range(training_steps):
            flow_params, xi_params_max_norm, opt_state, opt_state_xi, loss, xi_grads, xi_updates, conditional_lp, theta_0, x, x_noiseless, noise, EIG = update_pce(
                flow_params, xi_params_max_norm, next(prng_seq), opt_state, opt_state_xi, N=N, M=M, designs=d_sim, 
            )
            
            if jnp.any(jnp.isnan(xi_grads['xi'])):
                print("Gradients contain NaNs. Breaking out of loop.")
                break
            
            # Calculate the KL-div before updating designs
            like_log_probs = distrax.MultivariateNormalDiag(x_noiseless, noise).log_prob(x)
            kl_div = jnp.mean(like_log_probs - conditional_lp)
            
            # Setting bounds on the designs
            xi_params_max_norm['xi'] = jnp.clip(
                xi_params_max_norm['xi'], 
                a_min=jnp.divide(design_min, scale_factor), 
                a_max=jnp.divide(design_max, scale_factor)
                )
            
            # Unnormalize to use for simulator params
            xi_params['xi'] = jnp.multiply(xi_params_max_norm['xi'], scale_factor)

            # Update d_sim vector for new simulations
            if jnp.size(d) == 0:
                d_sim = xi_params['xi']
            else:
                d_sim = jnp.concatenate((d, xi_params['xi']), axis=1)

            # Saving contents to file
            print(f"STEP: {step:5d}; d_sim: {d_sim}; Xi: {xi_params['xi']}; \
            Xi Updates: {xi_updates['xi']}; Loss: {loss}; EIG: {EIG}; KL Div: {kl_div}")

            # wandb.log({"loss": loss, "xi": xi_params['xi'], "xi_grads": xi_grads['xi'], "kl_divs": kl_div, "EIG": EIG})

        # ---------------------------------
        # Approximate the posterior using VI
        # prior = make_lin_reg_prior()

        # # Evaluate the log-prior for all prior samples
        # prior_samples, prior_log_prob = prior.sample_and_log_prob(seed=next(prng_seq), sample_shape=(vi_samples))

        # xi = jnp.broadcast_to(xi_params['xi'], (vi_samples, xi_params['xi'].shape[-1]))

        # # Simulate data using the prior
        # x, prior_samples, _, _ = sim_linear_data_vmap(d_sim, vi_samples, next(prng_seq))
        # # TODO: Figure out prior_samples shape and simulate the correct response
        # log_likelihoods = log_prob.apply(flow_params, x, prior_samples, xi)

        # vi_params = vi_log_prob.init(
        #     next(prng_seq),
        #     np.zeros((1, *theta_shape)),
        # )

        # vi_optimizer = optax.adam(learning_rate)

        # @jax.jit
        # def vi_objective(vi_params, prior_samples, likelihood_log_probs, prior_log_probs):
        #     log_q = vi_log_prob.apply(vi_params, prior_samples)
        #     log_joint = likelihood_log_probs + prior_log_probs
        #     return -jnp.mean(log_joint - log_q)

        # @jax.jit
        # def vi_update(params: hk.Params,
        #             opt_state: OptState,
        #             prior_samples,
        #             likelihood_log_probs,
        #             prior_log_probs,
        #             ) -> Tuple[hk.Params, OptState]:
        #     """Single SGD update step of the VI posterior."""
        #     grads = jax.grad(vi_objective)(
        #         vi_params, prior_samples, likelihood_log_probs, prior_log_probs)
        #     updates, new_opt_state = vi_optimizer.update(grads, opt_state)
        #     new_params = optax.apply_updates(params, updates)
        #     return new_params, new_opt_state

        # vi_opt_state = vi_optimizer.init(vi_params)

        # for i in range(10):
        #     vi_params, vi_opt_state = vi_update(
        #         vi_params, vi_opt_state, prior_samples, log_likelihoods, prior_log_prob)

        # # Sample from the optimized variational family to approximate the posterior
        # # TODO: Implement sample function to use for evaluation metrics
        # shift = jnp.mean(prior_samples)
        # scale = jnp.std(prior_samples)
        # # posterior_samples = vi_sample.apply(
        # #     vi_params, next(prng_seq), num_samples=1000, shift=shift, scale=scale
        # #     )
        return flow_params, xi_params
        
    # Define the number of models to train in parallel
    num_models = 10

    # Initialize a list of random keys to use for each model
    # rng_keys = jrandom.split(jax.random.PRNGKey(0), num_models)
    rng_keys = [i for i in range(10)]

    delattr(cfg, 'seed')

    # Define the number of processes to use for training
    num_processes = -1

    # Parallelize the training of the models using joblib
    flow_params, xi_params = joblib.Parallel(n_jobs=num_processes)(
        joblib.delayed(train_single_model)(rng_key)
        for rng_key in rng_keys
    )



    return xi_params


if __name__=="__main__":
    main()