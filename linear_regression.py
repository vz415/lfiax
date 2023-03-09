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

        self.seed = self.cfg.seed

        if not self.cfg.designs.d:
            self.d = jnp.array([])
            self.xi = jnp.array([self.cfg.designs.xi])
            self.d_sim = self.xi # jnp.array([self.cfg.designs.xi])
        else:
            self.d = jnp.array(self.cfg.designs.d)
            self.xi = jnp.array([self.cfg.designs.xi])
            self.d_sim = jnp.concatenate((self.d, self.xi), axis=0)

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
        # logf, writer = self._init_logging()
        tic = time.time()

        @partial(jax.jit, static_argnums=[5,6])
        def update_pce(
            flow_params: hk.Params, xi_params: hk.Params, prng_key: PRNGKey, opt_state: OptState, opt_state_xi: OptState, N: int, M: int, designs: Array,
        ) -> Tuple[hk.Params, OptState]:
            """Single SGD update step."""
            log_prob_fun = lambda params, x, theta, xi: self.log_prob.apply(
                params, x, theta, xi)
            
            (loss, (conditional_lp, theta_0, x_noiseless, noise)), grads = jax.value_and_grad(
                lfi_pce_eig_scan, argnums=[0,1], has_aux=True)(
                flow_params, xi_params, prng_key, log_prob_fun, designs, N=N, M=M)
            
            updates, new_opt_state = optimizer.update(grads[0], opt_state)
            xi_updates, xi_new_opt_state = optimizer2.update(grads[1], opt_state_xi)

            new_params = optax.apply_updates(flow_params, updates)
            new_xi_params = optax.apply_updates(xi_params, xi_updates)

            return new_params, new_xi_params, new_opt_state, xi_new_opt_state, loss, grads[1], xi_updates, conditional_lp, theta_0, x_noiseless, noise #updates['xi']
        
         # Initialize the params
        prng_seq = hk.PRNGSequence(self.seed)
        params = self.log_prob.init(
            next(prng_seq),
            np.zeros((1, *self.EVENT_SHAPE)),
            np.zeros((1, *self.theta_shape)),
            np.zeros((1, *self.xi_shape)),
        )

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        # This could be initialized by a distribution of designs!
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

        params['xi'] = self.xi

        # Making its own unique haiku dicitonary for now but can change
        xi_params = {key: value for key, value in params.items() if key == 'xi'}

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
            
        opt_state_xi = optimizer2.init(xi_params)
        
        flow_params = {key: value for key, value in params.items() if key != 'xi'}

        for step in range(self.training_steps):
            if self.xi_optimizer == "Stupid_Adam":
                flow_params, xi_params, opt_state, _, loss, xi_grads, xi_updates, conditional_lp, theta_0, x_noiseless, noise = update_pce(
                    flow_params, xi_params, next(prng_seq), opt_state, opt_state_xi, N=self.N, M=self.M, designs=self.d_sim, 
                )
                xi_updates['xi'] = xi_updates['xi'] * (self.xi_lr_end ** (step / self.training_steps))
            else:
                flow_params, xi_params, opt_state, opt_state_xi, loss, xi_grads, xi_updates, conditional_lp, theta_0, x_noiseless, noise = update_pce(
                    flow_params, xi_params, next(prng_seq), opt_state, opt_state_xi, N=self.N, M=self.M, designs=self.d_sim, 
                )
            
            # Calculate the KL-div before updating designs
            # TODO: Make sure `MultivariateNormalDiag` is right distribution & implementation
            log_probs = distrax.MultivariateNormalDiag(x_noiseless, noise).log_prob(x_noiseless)
            kl_div = abs(jnp.sum(log_probs - conditional_lp))
            
            # Setting bounds on the designs
            xi_params['xi'] = jnp.clip(xi_params['xi'], a_min=-10., a_max=10.)

            # Update d_sim vector
            if jnp.size(self.d) == 0:
                self.d_sim = xi_params['xi']
            else:
                self.d_sim = jnp.concatenate((self.d, xi_params['xi']), axis=0)
            
            run_time = time.time()-tic

            # Saving contents to file
            print(f"STEP: {step:5d}; Xi: {xi_params['xi']}; Xi Updates: {xi_updates['xi']}; Loss: {loss}; KL Div: {kl_div}; ")

            wandb.log({"loss": loss, "xi": xi_params['xi'], "xi_grads": xi_grads['xi'], "kl_divs": kl_div})

            # writer.writerow({
            #     'step': step, 
            #     'xi': float(self.xi),
            #     'xi_grads': float(self.xi_grads),
            #     'loss': float(self.loss),
            #     'kl_div': float(kl_div),
            #     'time':float(run_time)
            # })
            # logf.flush()
            # self.save('latest')


    def save(self, tag='latest'):
        pass
        path = HydraConfig.get().runtime.output_dir
        # Creating a dictionary from the values since there is pickling error
        # when trying to pickle the entire object
        save_dict = {
            "xi": self.xi,
            "xi_grads": self.xi_grads,
            "loss": self.loss,
        }
        with open(path, 'wb') as f:
            pkl.dump(save_dict, f)

    def _init_logging(self):
        '''
        This function writes a csv to the working directory.
        '''
        pass
        path = os.path.join(HydraConfig.get().runtime.output_dir, 'log.csv')
        logf = open(path, 'a') 
        fieldnames = ['step', 'xi', 'xi_grads', 'loss', 'time']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat(path).st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


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
