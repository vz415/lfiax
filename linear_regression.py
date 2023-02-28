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
        else:
            self.d = jnp.array(self.cfg.designs.d)
        self.xi = jnp.array([self.cfg.designs.xi])
        self.d_sim = jnp.concatenate((self.d, self.xi), axis=0)

        # Bunch of event shapes needed for various functions
        len_xi = len(self.xi)
        self.xi_shape = (len_xi,)
        self.theta_shape = (2,)
        self.EVENT_SHAPE = (len(self.d_sim),)
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
        self.training_steps = self.cfg.optimization_params.training_steps
        self.lambda_ = self.cfg.optimization_params.lambda_

        @jax.jit
        def lagrange_weight_schedule_decay(iteration, decay_rate=0.99):
            """
            Returns the Lagrange multiplier weight for a given iteration using an exponential decay.
            """
            return decay_rate ** iteration

        @jax.jit
        def lagrange_weight_schedule_sigmoid(iteration, max_iter, slope=10):
            """
            Returns the Lagrange multiplier weight for a given iteration using a sigmoidal schedule.
            """
            weight = 1 - (1 / (1 + jnp.exp(-slope * (iteration - max_iter/2) / max_iter)))
            return weight


        if self.cfg.lagrange_weight_sch.decay and self.cfg.lagrange_weight_sch.sigmoid:
            raise AssertionError ("Make sure only one weight schedule is selected")
        elif self.cfg.lagrange_weight_sch.decay:
            self.weight_schedule = lagrange_weight_schedule_decay
        elif self.cfg.lagrange_weight_sch.sigmoid:
            self.weight_schedule = lagrange_weight_schedule_sigmoid
            self.sigmoid_slope = self.cfg.lagrange_weight_sch.sigmoid_params.slope
        else:
            self.weight_schedule = None


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

        @partial(jax.jit, static_argnums=[3,4])
        def update_pce(
            params: hk.Params, prng_key: PRNGKey, opt_state: OptState, N: int, M: int, designs: Array, lambda_: float,
        ) -> Tuple[hk.Params, OptState]:
            """Single SGD update step."""
            log_prob_fun = lambda params, x, theta, xi: self.log_prob.apply(
                params, x, theta, xi)
            loss, grads = jax.value_and_grad(lfi_pce_eig_scan)(
                params, prng_key, log_prob_fun, designs, N=N, M=M, lambda_=lambda_)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss, grads['xi']
        
         # Initialize the params
        prng_seq = hk.PRNGSequence(self.seed)
        params = self.log_prob.init(
            next(prng_seq),
            np.zeros((1, *self.EVENT_SHAPE)),
            np.zeros((1, *self.theta_shape)),
            np.zeros((1, *self.xi_shape)),
        )
        # This could be initialized by a distribution of designs!
        params['xi'] = self.xi

        optimizer = optax.adam(self.learning_rate)

        opt_state = optimizer.init(params)

        for step in range(self.training_steps):
            params, opt_state, loss, xi_grads = update_pce(
                params, next(prng_seq), opt_state, N=self.N, M=self.M, designs=self.d_sim, lambda_=self.lambda_,
            )
            
            # Update d_sim vector
            self.d_sim = jnp.concatenate((self.d, params['xi']), axis=0)
            
            if self.weight_schedule:
                if self.cfg.lagrange_weight_sch.decay:
                    self.lambda_ = self.weight_schedule(step)
                elif self.cfg.lagrange_weight_sch.sigmoid:
                    self.lambda_ = self.weight_schedule(step, self.training_steps, slope=self.sigmoid_slope)
            
            run_time = time.time()-tic

            # Saving contents to file
            print(f"STEP: {step:5d}; Xi: {params['xi']}; Xi Grads: {xi_grads}; \
             Loss: {loss}")

            self.xi = params['xi']
            self.xi_grads = xi_grads
            self.loss = loss

            wandb.log({"loss": loss, "xi": params["xi"], "xi_grads": xi_grads})

            # writer.writerow({
            #    'step': step, 
            #    'xi': float(self.xi),
            #    'xi_grads': float(self.xi_grads),
            #    'loss': float(self.loss),
            #    'time':float(run_time)
            # })
            # logf.flush()
            # self.save('latest')



    def save(self, tag='latest'):
        pass
        # path = os.path.join(self.work_dir, f'{tag}.pkl')
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
        #path = os.path.join(self.work_dir, 'log.csv')
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
