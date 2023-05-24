import omegaconf
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb
import os
import csv, time
import pickle as pkl
import math
import random

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
from lfiax.utils.oed_losses import snpe_c
from lfiax.utils.simulators import sim_linear_data_vmap_theta

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

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
            )
        wandb.config.update(wandb.config)
        wandb.init(
            entity=self.cfg.wandb.entity, 
            project=self.cfg.wandb.project, 
            config=wandb.config
            )

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        current_time = time.localtime()
        current_time_str = f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}.{current_time.tm_hour:02d}.{current_time.tm_min:02d}"
        
        eig_lambda_str = str(cfg.optimization_params.eig_lambda).replace(".", "-")
        file_name = f"eig_lambda_{eig_lambda_str}"
        self.subdir = os.path.join(os.getcwd(), "icml_linear", 'snpe_pce_lin_reg', file_name, str(cfg.designs.num_xi), str(cfg.seed), current_time_str)
        os.makedirs(self.subdir, exist_ok=True)

        self.seed = self.cfg.seed
        rng = jrandom.PRNGKey(self.seed)
        
        if self.cfg.designs.num_xi is not None:
            if self.cfg.designs.d is None:
                self.d = jnp.array([])
                self.xi = jrandom.uniform(rng, shape=(self.cfg.designs.num_xi,), minval=-10, maxval=10)
                self.d_sim = self.xi
            else:
                self.d = jnp.array([self.cfg.designs.d])
                self.xi = jrandom.uniform(rng, shape=(self.cfg.designs.num_xi,), minval=1e-6, maxval=1e6)
                self.d_sim = jnp.concatenate((self.d, self.xi), axis=1)
        else:
            if self.cfg.designs.d is None:
                self.d = jnp.array([])
                self.xi = jnp.array([self.cfg.designs.xi])
                self.d_sim = self.xi
            elif self.cfg.designs.xi is None:
                self.d = jnp.array([self.cfg.designs.d])
                self.xi = jnp.array([])
                self.d_sim = self.d
            else:
                self.d = jnp.array([self.cfg.designs.d])
                self.xi = jnp.array([self.cfg.designs.xi])
                self.d_sim = jnp.concatenate((self.d, self.xi), axis=1)

        # Bunch of event shapes needed for various functions
        len_xi = self.xi.shape[-1]
        self.xi_shape = (len_xi,)
        self.theta_shape = (2,)
        self.EVENT_SHAPE = (self.d_sim.shape[-1],)
        EVENT_DIM = self.cfg.param_shapes.event_dim

        # contrastive sampling parameters
        self.M = self.cfg.contrastive_sampling.M
        self.N = self.cfg.contrastive_sampling.N

        # likelihood flow's params
        flow_num_layers = self.cfg.flow_params.num_layers
        mlp_num_layers = self.cfg.flow_params.mlp_num_layers
        hidden_size = self.cfg.flow_params.mlp_hidden_size
        num_bins = self.cfg.flow_params.num_bins

        # Optimization parameters
        self.learning_rate = self.cfg.optimization_params.learning_rate
        self.xi_lr_init = self.cfg.optimization_params.xi_learning_rate
        self.training_steps = self.cfg.optimization_params.training_steps
        self.xi_optimizer = self.cfg.optimization_params.xi_optimizer
        self.xi_scheduler = self.cfg.optimization_params.xi_scheduler
        self.xi_lr_end = self.cfg.optimization_params.xi_lr_end
        self.eig_lambda = self.cfg.optimization_params.eig_lambda

        # Scheduler to use
        if self.xi_scheduler == "None":
            self.schedule = self.xi_lr_init
        
        @hk.without_apply_rng
        @hk.transform
        def posterior_log_prob(theta: Array, x: Array, xi: Array) -> Array:
            theta_scaled = standard_scale(theta)
            model = make_nsf(
                event_shape=self.theta_shape,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=True,
                use_resnet=True,
                conditional=True
            )
            return model.log_prob(theta_scaled, x, xi)

        self.post_log_prob = posterior_log_prob

    def run(self) -> Callable:
        logf, writer = self._init_logging()

        @partial(jax.jit, static_argnums=[5,8,9,10])
        def update_snpe_pce(
            post_params: hk.Params, xi_params: hk.Params, prng_key: PRNGKey, \
            opt_state: OptState, opt_state_xi: OptState, prior: Callable, 
            scaled_x: Array, theta_0: Array, N: int, M: int, lam: float,
        ) -> Tuple[hk.Params, OptState]:
            """Single SGD update step."""
            post_log_prob_fun = lambda params, x, theta, xi: self.post_log_prob.apply(
                params, x, theta, xi)
            
            (loss, (conditional_lp, EIG)), grads = jax.value_and_grad(
                snpe_c, argnums=[0,1], has_aux=True)(
                post_params, xi_params, prng_key, prior, scaled_x, theta_0,
                post_log_prob_fun, N=N, M=M, lam=lam
                )
            
            updates, new_opt_state = optimizer.update(grads[0], opt_state)
            xi_updates, xi_new_opt_state = optimizer2.update(grads[1], opt_state_xi)

            new_params = optax.apply_updates(post_params, updates)
            new_xi_params = optax.apply_updates(xi_params, xi_updates)
            
            return new_params, new_xi_params, new_opt_state, xi_new_opt_state, loss, grads[1], xi_updates, conditional_lp, EIG
        
        # Initialize the net's params
        prng_seq = hk.PRNGSequence(self.seed)
        post_params = self.post_log_prob.init(
            next(prng_seq),
            np.zeros((1, *self.theta_shape)),
            np.zeros((1, *self.EVENT_SHAPE)),
            np.zeros((1, *self.xi_shape)),
        )
        
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(post_params)

        if self.xi_optimizer == "Adam":
            optimizer2 = optax.adam(learning_rate=self.schedule)
        elif self.xi_optimizer == "SGD":
            optimizer2 = optax.sgd(learning_rate=self.schedule)
        
        # This could be initialized by a distribution of designs!
        post_params['xi'] = self.xi
        xi_params = {key: value for key, value in post_params.items() if key == 'xi'}
        
        # Normalize xi values for optimizer
        design_min = -10.
        design_max = 10.
        scale_factor = float(jnp.max(jnp.array([jnp.abs(design_min), jnp.abs(design_max)])))
        xi_params_max_norm = {}
        xi_params_max_norm['xi'] = jnp.divide(xi_params['xi'], scale_factor)
        # TODO: try normal scaling the xi_params to get more consistent training
        # xi_params_scaled = (xi_params['xi'] - jnp.mean(xi_params['xi'])) / (jnp.std(xi_params['xi']) + 1e-10)

        opt_state_xi = optimizer2.init(xi_params_max_norm)
        post_params = {key: value for key, value in post_params.items() if key != 'xi'}
        
        priors = make_lin_reg_prior()
        
        for step in range(self.training_steps):
            tic = time.time()
            # get priors and simulate a data point
            theta_0 = priors.sample(seed=next(prng_seq), sample_shape=(self.N,))
            # breakpoint()
            x, _, _  = sim_linear_data_vmap_theta(self.d_sim, theta_0, next(prng_seq))
            
            scaled_x = standard_scale(x)
            x_mean, x_std = jnp.mean(x), jnp.std(x) + 1e-10
            simulate_time = time.time() - tic
            tic = time.time()

            post_params, xi_params_max_norm, opt_state, opt_state_xi, loss, xi_grads, xi_updates, conditional_lp, EIG = update_snpe_pce(
                post_params, xi_params_max_norm, next(prng_seq), opt_state, opt_state_xi, priors, scaled_x, theta_0=theta_0.squeeze(), N=self.N, M=self.M, lam=self.eig_lambda
            )
            
            if jnp.any(jnp.isnan(xi_grads['xi'])):
                print("Gradients contain NaNs. Breaking out of loop.")
                break
            
            # Setting bounds on the designs
            xi_params_max_norm['xi'] = jnp.clip(
                xi_params_max_norm['xi'], 
                a_min=jnp.divide(design_min, scale_factor), 
                a_max=jnp.divide(design_max, scale_factor)
                )
            
            # Unnormalize to use for simulator params
            xi_params['xi'] = jnp.multiply(xi_params_max_norm['xi'], scale_factor)

            # Update d_sim vector for new simulations
            if jnp.size(self.d) == 0:
                self.d_sim = xi_params['xi']
            elif jnp.size(self.xi) == 0:
                self.d_sim =self.d_sim
            else:
                self.d_sim = jnp.concatenate((self.d, xi_params['xi']), axis=1)
            
            inference_time = time.time()-tic

            # Saving contents to file
            # print(f"STEP: {step:5d}; d_sim: {float(self.d_sim):.5f}; Xi: {float(xi_params['xi']):.5f}; \
            # Xi Updates: {float(xi_updates['xi']):.6f}; Loss: {float(loss):.5f}; EIG: {float(EIG):.5f}; Inference Time: {inference_time:.5f} \
            print(f"STEP: {step:5d}; Xi: {xi_params['xi']}; \
            Xi Updates: {xi_updates['xi']}; Loss: {float(loss):.5f}; EIG: {float(EIG):.5f}; Inference Time: {inference_time:.5f} \
            Simulate Time: {simulate_time:.5f}")

            writer.writerow({
                'STEP': step, 
                'Xi': xi_params['xi'],
                'Loss': loss,
                'EIG': EIG,
                'inference_time':float(inference_time),
            })
            logf.flush()

            wandb.log({"loss": loss, "xi": xi_params['xi'], "xi_grads": xi_grads['xi'], "EIG": EIG})

        # Create a dictionary to store the objects.

        # objects = {
        #     'flow_params': jax.device_get(post_params),
        #     'xi_params': jax.device_get(xi_params),
        # }

        # Save the objects.
        # with open("SNPE_best_100D_params.pkl", "wb") as f:
        #     pkl.dump(objects, f)

        
    def _init_logging(self):
        path = os.path.join(self.subdir, 'log.csv')
        logf = open(path, 'a') 
        fieldnames = ['STEP', 'Xi', 'Loss', 'EIG', 'inference_time']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat(path).st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer

from linear_regression_snpe import Workspace as W

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    fname = os.getcwd() + '/latest.pt'
    if os.path.exists(fname):
        #TODO: Test this portion of the code
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = pkl.load(f)
        print(f"STEP: {workspace.step:5d}; Xi: {workspace.xi};\
             Xi Grads: {workspace.xi_grads}; Loss: {workspace.loss}")
    else:
        workspace = W(cfg)

    workspace.run()


if __name__ == "__main__":
    main()
