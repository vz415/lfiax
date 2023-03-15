import omegaconf
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb
from collections import deque
import os
import csv, time
import pickle as pkl
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


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        # wandb.config = omegaconf.OmegaConf.to_container(
        #     cfg, resolve=True, throw_on_missing=True
        #     )
        # wandb.config.update(wandb.config)
        # wandb.init(
        #     entity=self.cfg.wandb.entity, 
        #     project=self.cfg.wandb.project, 
        #     config=wandb.config
        #     )

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.seed = self.cfg.seed
        
        if self.cfg.designs.d is None:
            self.d = jnp.array([])
            self.xi = jnp.array([self.cfg.designs.xi])
            self.d_sim = self.xi # jnp.array([self.cfg.designs.xi])
        else:
            # Should these have another dimension to make later processing easier?
            self.d = jnp.array([self.cfg.designs.d])
            self.xi = jnp.array([self.cfg.designs.xi])
            self.d_sim = jnp.concatenate((self.d, self.xi), axis=1)

        # Bunch of event shapes needed for various functions
        # len_xi = len(self.xi)
        len_xi = self.xi.shape[-1]
        self.xi_shape = (len_xi,)
        self.theta_shape = (2,)
        self.EVENT_SHAPE = (self.d_sim.shape[-1],)
        EVENT_DIM = self.cfg.param_shapes.event_dim

        # contrastive sampling parameters
        self.M = self.cfg.contrastive_sampling.M
        self.N = self.cfg.contrastive_sampling.N

        # flow's params
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
        self.xi_lr_end = 1e-4


        @hk.without_apply_rng
        @hk.transform
        def log_prob(data: Array, theta: Array, xi: Array) -> Array:
            shift = data.mean(axis=0)
            scale = data.std(axis=0) + 1e-14
            # TODO: Pass more nsf parameters from config.yaml
            model = make_nsf(
                event_shape=self.EVENT_SHAPE,
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

        self.log_prob = log_prob


    def run(self) -> Callable:
        tic = time.time()

        @partial(jax.jit, static_argnums=[5,6,7])
        def update_pce(
            flow_params: hk.Params, xi_params: hk.Params, prng_key: PRNGKey, \
            opt_state: OptState, opt_state_xi: OptState, N: int, M: int, \
            scale_factor: int, designs: Array,
        ) -> Tuple[hk.Params, OptState]:
            """Single SGD update step."""
            log_prob_fun = lambda params, x, theta, xi: self.log_prob.apply(
                params, x, theta, xi)
            
            (loss, (conditional_lp, theta_0, x_noiseless, noise, EIG)), grads = jax.value_and_grad(
                lfi_pce_eig_scan, argnums=[0,1], has_aux=True)(
                flow_params, xi_params, prng_key, log_prob_fun, designs, scale_factor, N=N, M=M)
            
            updates, new_opt_state = optimizer.update(grads[0], opt_state)
            xi_updates, xi_new_opt_state = optimizer2.update(grads[1], opt_state_xi)

            new_params = optax.apply_updates(flow_params, updates)
            new_xi_params = optax.apply_updates(xi_params, xi_updates)

            return new_params, new_xi_params, new_opt_state, xi_new_opt_state, loss, grads[1], xi_updates, conditional_lp, theta_0, x_noiseless, noise, EIG
        
         # Initialize the net's params
        prng_seq = hk.PRNGSequence(self.seed)
        params = self.log_prob.init(
            next(prng_seq),
            np.zeros((1, *self.EVENT_SHAPE)),
            np.zeros((1, *self.theta_shape)),
            np.zeros((1, *self.xi_shape)),
        )

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        if self.xi_scheduler == "None":
            schedule = self.xi_lr_init
        elif self.xi_scheduler == "Linear":
            schedule = optax.linear_schedule(self.xi_lr_init, self.xi_lr_end, transition_steps=self.training_steps)
        elif self.xi_scheduler == "Exponential":
            schedule = optax.exponential_decay(
                init_value=self.xi_lr_init,
                transition_steps=self.training_steps,
                decay_rate=(self.xi_lr_end / self.xi_lr_init) ** (1 / num_epochs),
                staircase=False
            )
        else:
            raise AssertionError("Specified unsupported scheduler.")

        if self.xi_optimizer == "Stupid_Adam":
            optimizer2 = optax.adam(learning_rate=self.xi_lr_init)
        elif self.xi_optimizer == "Adam":
            optimizer2 = optax.adam(learning_rate=schedule)
        elif self.xi_optimizer == "SGD":
            optimizer2 = optax.sgd(learning_rate=schedule)
        elif self.xi_optimizer == "Yogi":
            optimizer2 = optax.yogi(learning_rate=schedule)
        elif self.xi_optimizer == "AdaBelief":
            optimizer2 = optax.adabelief(learning_rate=schedule)
        
        # This could be initialized by a distribution of designs!
        # Making xi its own unique haiku dicitonary for now but can change
        params['xi'] = self.xi
        xi_params = {key: value for key, value in params.items() if key == 'xi'}
        
        # Normalize xi values for optimizer
        design_min = -10.
        design_max = 10.
        scale_factor = float(jnp.max(jnp.array([jnp.abs(design_min), jnp.abs(design_max)])))
        # xi_params_scaled = (xi_params['xi'] - jnp.mean(xi_params['xi'])) / jnp.std(xi_params['xi'])
        xi_params_max_norm = {}
        xi_params_max_norm['xi'] = jnp.divide(xi_params['xi'], scale_factor)

        # TODO: swap in scaled xi params to optimization routine.
        # opt_state_xi = optimizer2.init(xi_params)
        opt_state_xi = optimizer2.init(xi_params_max_norm)
        flow_params = {key: value for key, value in params.items() if key != 'xi'}

        for step in range(self.training_steps):
            if self.xi_optimizer == "Stupid_Adam":
                flow_params, xi_params_max_norm, opt_state, _, loss, xi_grads, xi_updates, conditional_lp, theta_0, x_noiseless, noise, EIG = update_pce(
                    flow_params, xi_params_max_norm, next(prng_seq), opt_state, opt_state_xi, N=self.N, M=self.M, designs=self.d_sim, 
                )
                # This shouldn't work, but, somehow, it does (jk it doesn't)
                print(f"opt_state_xi: {opt_state_xi}")
                xi_updates['xi'] = xi_updates['xi'] * (self.xi_lr_end ** (step / self.training_steps))
            else:
                # flow_params, xi_params, opt_state, opt_state_xi, loss, xi_grads, xi_updates, conditional_lp, theta_0, x_noiseless, noise, EIG = update_pce(
                #     flow_params, xi_params, next(prng_seq), opt_state, opt_state_xi, N=self.N, M=self.M, designs=self.d_sim, 
                # )
                flow_params, xi_params_max_norm, opt_state, opt_state_xi, loss, xi_grads, xi_updates, conditional_lp, theta_0, x_noiseless, noise, EIG = update_pce(
                    flow_params, xi_params_max_norm, next(prng_seq), opt_state, opt_state_xi, N=self.N, M=self.M, scale_factor=scale_factor, designs=self.d_sim, 
                )
                print(f"opt_state_xi: {opt_state_xi}")
            
            # Calculate the KL-div before updating designs
            # TODO: Make sure `MultivariateNormalDiag` is right distribution & implementation
            log_probs = distrax.MultivariateNormalDiag(x_noiseless, noise).log_prob(x_noiseless)
            kl_div = abs(jnp.sum(log_probs - conditional_lp))
            
            # Setting bounds on the designs
            # xi_params_max_norm['xi'] = jnp.clip(xi_params['xi'], a_min=-10., a_max=10.)
            # TODO: Fix this error.
            xi_params_max_norm['xi'] = jnp.clip(
                xi_params_max_norm['xi'], 
                a_min=jnp.divide(design_min, scale_factor), 
                a_max=jnp.divide(design_max, scale_factor)
                )
            # Unnormalize to use for simulator params
            xi_params['xi'] = jnp.multiply(xi_params_max_norm['xi'], scale_factor)

            # Update d_sim vector
            if jnp.size(self.d) == 0:
                self.d_sim = xi_params['xi']
            else:
                self.d_sim = jnp.concatenate((self.d, xi_params['xi']), axis=1)
            
            run_time = time.time()-tic

            # Saving contents to file
            print(f"STEP: {step:5d}; d_sim: {self.d_sim}; Xi: {xi_params['xi']}; Xi Updates: {xi_updates['xi']}; Loss: {loss}; EIG: {EIG}; KL Div: {kl_div}; ")

            # wandb.log({"loss": loss, "xi": xi_params['xi'], "xi_grads": xi_grads['xi'], "kl_divs": kl_div, "EIG": EIG})


from linear_regression import Workspace as W

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
