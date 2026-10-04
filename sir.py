import omegaconf
import hydra
import wandb
import os
import csv
import time
import itertools
import pickle as pkl
import warnings
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrandom
import blackjax

import torch

import numpy as np
from functools import partial
import optax
import distrax
import haiku as hk

from sbi.diagnostics.lc2st import LC2ST

from lfiax.flows.nsf import make_nsf
from lfiax.utils.oed_losses import lf_pce_design_dist_sir
from lfiax.utils.simulators import simulate_sir, sample_lognormal_with_log_probs, lognormal_log_prob, collect_sufficient_sde_samples_prior
from lfiax.utils.utils import run_mcmc, run_mcmc_smc, shuffle_samples, split_data_for_validation_jax, prior_to_standard_normal, prior_lp_logdetjac
from lfiax.utils.sbi_losses import kl_sbc_loss_fn_general, kl_loss_fn_general
from lfiax.utils.sir_utils import LossSmoother, reduce_on_plateau
from lfiax.utils.update_funs import update_pce
from lfiax.utils.design_baselines import select_baseline_design, validate_baseline_config


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
Batch = Mapping[str, np.ndarray]
OptState = Any

@jax.jit
def normalize_xi_to_gaussian(x):
    scaled_x = x / 100.0
    # Apply probit function (inverse normal CDF)
    normalized_x = jax.scipy.stats.norm.ppf(scaled_x)
    return normalized_x

@jax.jit
def inverse_normalize_xi(normalized_x):
    # Inverse of the probit function (normal CDF)
    scaled_x = jax.scipy.stats.norm.cdf(normalized_x)
    # Rescale back to original range
    x = scaled_x * 100.0
    return x

def compute_average_norm(grads):
    norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
    flat_norms = jax.tree_util.tree_leaves(norms)
    average_norm = jnp.mean(jnp.array(flat_norms))
    return average_norm

def check_for_nans(param_dict):
    def is_nan(x):
        return jnp.any(jnp.isnan(x))
    nan_map = jax.tree_util.tree_map(is_nan, param_dict)
    return jax.tree_util.tree_reduce(lambda x, y: x or y, nan_map, initializer=False)

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.wandb.use_wandb:
            wandb.config = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
                )
            wandb.config.update(wandb.config)
            wandb.init(
                entity=self.cfg.wandb.entity,
                project=self.cfg.wandb.project,
                config=wandb.config
                )
            # Defining the axes for each part
            wandb.define_metric("post/step")
            wandb.define_metric("post/*", step_metric="post/step")

        # Number of design rounds to perform optimization
        self.design_rounds = self.cfg.experiment.design_rounds
        self.refine_rounds = self.cfg.experiment.refine_rounds
        self.posterior_pool_size = self.cfg.experiment.posterior_pool_size
        self.sbi_train_steps = self.cfg.experiment.sbi_train_steps
        self.sir_type = self.cfg.experiment.sir_type
        self.device = self.cfg.experiment.device
        self.hpc = self.cfg.experiment.hpc
        self.debug = self.cfg.experiment.debug
        baseline_cfg = self.cfg.get("baseline", {})
        self.design_policy = baseline_cfg.get("design_policy", "optimized")
        self.likelihood_objective = baseline_cfg.get("likelihood_objective", "infonce_lambda")
        early_stopping_cfg = baseline_cfg.get("early_stopping", {})
        self.baseline_early_stopping = early_stopping_cfg.get("enabled", True)
        self.baseline_early_stopping_patience = int(early_stopping_cfg.get("patience", 10))
        self.baseline_early_stopping_scale = float(early_stopping_cfg.get("scale", 1e-3))
        validate_baseline_config(self.design_policy, self.likelihood_objective)

        if self.hpc:
            self.work_dir = "/pub/vzaballa/lfiax_data"
            print(f'workspace: {self.work_dir}')

            current_time = time.localtime()
            current_time_str = f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}.{current_time.tm_hour:02d}.{current_time.tm_min:02d}"

            eig_lambda_str = str(cfg.optimization_params.eig_lambda).replace(".", "-")
            file_name = self.sir_type
            path_parts = [self.work_dir, "sir", file_name]
            if self.design_policy != "optimized":
                path_parts.extend([self.design_policy, self.likelihood_objective])
            path_parts.extend([str(cfg.seed), current_time_str])
            self.subdir = os.path.join(*path_parts)
            os.makedirs(self.subdir, exist_ok=True)
        else:
            # Work around since hydra logging is erring
            self.work_dir = os.getcwd()
            print(f'workspace: {self.work_dir}')

            current_time = time.localtime()
            current_time_str = f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}.{current_time.tm_hour:02d}.{current_time.tm_min:02d}"

            eig_lambda_str = str(cfg.optimization_params.eig_lambda).replace(".", "-")
            file_name = f"eig_lambda_{eig_lambda_str}"
            path_parts = [os.getcwd(), "sir", file_name]
            if self.design_policy != "optimized":
                path_parts.extend([self.design_policy, self.likelihood_objective])
            path_parts.extend([str(cfg.designs.num_xi), str(cfg.seed), current_time_str])
            self.subdir = os.path.join(*path_parts)
            os.makedirs(self.subdir, exist_ok=True)

        self.seed = self.cfg.seed

        self.xi_mu = self.cfg.designs.xi_mu
        self.xi_stddev = self.cfg.designs.xi_stddev
        self.d = None
        self.static_outputs_sbi = None
        self.use_design_dist = self.cfg.designs.use_design_dist

        # NOTE: Use prod likelihood for SIR (just 1D anyways)
        # Bunch of event shapes needed for various functions
        # len_xi = self.xi.shape[-1]
        # self.xi_shape = (len_xi,)
        self.xi_shape = (1,)
        self.theta_shape = (2,)
        # self.EVENT_SHAPE = (self.d_sim.shape[-1],)
        self.EVENT_SHAPE = (1,)
        EVENT_DIM = self.cfg.param_shapes.event_dim

        # contrastive sampling parameters
        self.M = self.cfg.contrastive_sampling.M
        self.N = self.cfg.contrastive_sampling.N
        if self.posterior_pool_size < self.N:
            raise ValueError(
                "experiment.posterior_pool_size must be greater than or equal to "
                "contrastive_sampling.N."
            )

        epig_cfg = self.cfg.get("epig", {})
        self.epig_enabled = bool(epig_cfg.get("enabled", False))
        self.epig_policy = epig_cfg.get("selection_policy", "top_k")
        self.epig_num_candidates = int(epig_cfg.get("num_candidates", 2 * self.N))
        self.epig_K = int(epig_cfg.get("num_particles", 8))
        self.epig_S = int(epig_cfg.get("num_joint_samples", 1))
        self.epig_antithetic_sampling = bool(
            epig_cfg.get("antithetic_sampling", False)
        )
        self.epig_dropout_crn = bool(epig_cfg.get("dropout_crn", False))
        self.epig_temperature = float(epig_cfg.get("softmax_temperature", 1.0))

        # likelihood flow's params
        flow_num_layers = self.cfg.flow_params.num_layers
        mlp_num_layers = self.cfg.flow_params.mlp_num_layers
        hidden_size = self.cfg.flow_params.mlp_hidden_size
        num_bins = self.cfg.flow_params.num_bins
        self.spline_range_min = float(
            self.cfg.flow_params.get("spline_range_min", 0.0)
        )
        self.spline_range_max = float(
            self.cfg.flow_params.get("spline_range_max", 1.0)
        )
        resnet = self.cfg.flow_params.resnet
        self.activation = self.cfg.flow_params.activation
        self.z_scale_theta = self.cfg.flow_params.z_scale_theta
        self.dropout_rate = self.cfg.flow_params.dropout_rate

        if self.epig_enabled:
            if self.epig_policy not in ("top_k", "softmax_corrected"):
                raise ValueError(
                    "epig.selection_policy must be 'top_k' or 'softmax_corrected'."
                )
            if self.epig_num_candidates < self.N:
                raise ValueError("epig.num_candidates must be greater than or equal to N.")
            if self.M >= self.N:
                raise ValueError("contrastive_sampling.M must be less than N.")
            if self.epig_K <= 0 or self.epig_S <= 0:
                raise ValueError("EPIG particle and joint-sample counts must be positive.")
            if self.epig_temperature <= 0:
                raise ValueError("epig.softmax_temperature must be positive.")
            if self.dropout_rate <= 0:
                raise ValueError("EPIG requires flow_params.dropout_rate to be positive.")

        # MCMC params
        self.num_adapt_steps = self.cfg.mcmc_params.num_adapt_steps
        self.num_mcmc_samples = self.cfg.mcmc_params.num_mcmc_samples

        # Optimization parameters
        self.learning_rate = self.cfg.optimization_params.learning_rate
        self.xi_lr_init = self.cfg.optimization_params.xi_learning_rate
        self.training_steps = self.cfg.optimization_params.training_steps
        self.refine_likelihood_rounds = self.cfg.optimization_params.refine_likelihood_rounds # noqa
        self.xi_optimizer = self.cfg.optimization_params.xi_optimizer
        self.xi_scheduler = self.cfg.optimization_params.xi_scheduler
        self.flow_beta2 = self.cfg.optimization_params.flow_beta2
        self.xi_beta2 = self.cfg.optimization_params.xi_beta2
        self.xi_lr_end = self.cfg.optimization_params.xi_lr_end
        self.eig_lambda = self.cfg.optimization_params.eig_lambda
        self.ewma_smoothing = self.cfg.optimization_params.ewma_smoothing
        self.coefficient = self.cfg.optimization_params.coefficient
        self.tau = self.cfg.optimization_params.tau
        self.grad_clip = self.cfg.optimization_params.grad_clip
        self.xi_grad_clip = self.cfg.optimization_params.xi_grad_clip
        self.end_sigma = self.cfg.optimization_params.end_sigma
        self.importance_sampling = self.cfg.optimization_params.imp_sampling
        self.y_scale = self.cfg.optimization_params.y_scale
        # Posterior optimization parameters
        self.post_num_layers = self.cfg.post_optimization.num_layers
        self.post_mlp_num_layers = self.cfg.post_optimization.mlp_num_layers
        self.post_hidden_size = self.cfg.post_optimization.mlp_hidden_size
        self.post_num_bins = self.cfg.post_optimization.num_bins
        self.post_resnet = self.cfg.post_optimization.resnet
        self.sbc_samples = self.cfg.post_optimization.sbc_samples
        self.sbc_lambda = self.cfg.post_optimization.sbc_lam
        self.vi_steps = self.cfg.post_optimization.vi_steps
        self.vi_samples = self.cfg.post_optimization.vi_samples

        # Scheduler params
        self.patience = self.cfg.xi_scheduler.patience
        self.reduce_factor = self.cfg.xi_scheduler.reduce_factor
        self.min_improvement = self.cfg.xi_scheduler.min_improvement
        self.cooldown = self.cfg.xi_scheduler.cooldown

        # Scheduler to use
        if self.xi_scheduler == "None":
            self.schedule = self.xi_lr_init
        elif self.xi_scheduler == "Custom":
            # LR scheduling items
            self.schedule = self.xi_lr_init
            self.previous_loss = float("inf")
            self.learning_rate = self.xi_lr_init
            self.momentum_term = 1.0
            self.loss_smoother = LossSmoother(beta=self.ewma_smoothing)
        else:
            raise AssertionError("Specified unsupported scheduler.")

        @jax.jit
        def log_trans_logdetjac(x):
            x = x + 1e-8
            jac_diag_o = jax.vmap(jax.grad(jnp.log))(x.reshape(-1))
            # Reshape jac_diag to match the original shape of x
            jac_diag = jac_diag_o.reshape(x.shape)
            # Compute log(abs(jac_diag))
            log_abs_jac = jnp.log(jnp.abs(jac_diag))
            # Sum across all dimensions except the first (batch dimension)
            logdetjac = jnp.sum(log_abs_jac, axis=tuple(range(1, log_abs_jac.ndim)))
            # Ensure the output has shape [N, 1]
            return logdetjac.reshape(-1, 1)

        @hk.transform
        def log_prob(x: Array, theta: Array, xi: Array) -> Array:
            """
            Likelihood isn't normalized. The data are all positive definite.
            Apply some transformation to the data to make it gaussian.
            """
            model = make_nsf(
                event_shape=self.EVENT_SHAPE,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=self.z_scale_theta,
                use_resnet=resnet,
                conditional=True,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                spline_range_min=self.spline_range_min,
                spline_range_max=self.spline_range_max,
            )
            # Since base is gaussian, transform from lognormal to normal
            log_x = jnp.log(x + 1e-8)
            if self.cfg.designs.norm_type == "ppf":
                norm_xi = normalize_xi_to_gaussian(xi)
            elif self.cfg.designs.norm_type == "log":
                norm_xi = jnp.log(xi)
            else:
                raise ValueError(f"Norm type {self.cfg.designs.norm_type} not recognized. And you better normalize.")
            norm_theta = prior_to_standard_normal(theta)
            lps = model.log_prob(log_x, norm_theta, norm_xi)
            logdetjac = log_trans_logdetjac(x)

            return lps - logdetjac.squeeze()

        self.log_prob = log_prob

        @hk.transform
        def sample_likelihood(
            prng_key: PRNGKey,
            num_samples: int,
            theta: Array,
            xi: Array,
        ) -> Array:
            """Sample raw infected counts from the dropout likelihood surrogate."""
            model = make_nsf(
                event_shape=self.EVENT_SHAPE,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=self.z_scale_theta,
                use_resnet=resnet,
                conditional=True,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                spline_range_min=self.spline_range_min,
                spline_range_max=self.spline_range_max,
            )
            norm_theta = prior_to_standard_normal(theta)
            if self.cfg.designs.norm_type == "ppf":
                norm_xi = normalize_xi_to_gaussian(xi)
            elif self.cfg.designs.norm_type == "log":
                norm_xi = jnp.log(xi)
            else:
                raise ValueError(
                    f"Norm type {self.cfg.designs.norm_type} not recognized."
                )
            log_x = model._sample_n(
                key=prng_key,
                n=num_samples,
                theta=norm_theta,
                xi=norm_xi,
            )
            return jnp.exp(log_x)

        self.sample_likelihood = sample_likelihood

        @hk.transform
        def sample_likelihood_from_base(
            base_sample: Array,
            theta: Array,
            xi: Array,
        ) -> Array:
            """Transform explicit standard-normal latents into infected counts."""
            model = make_nsf(
                event_shape=self.EVENT_SHAPE,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=self.z_scale_theta,
                use_resnet=resnet,
                conditional=True,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                spline_range_min=self.spline_range_min,
                spline_range_max=self.spline_range_max,
            )
            norm_theta = prior_to_standard_normal(theta)
            if self.cfg.designs.norm_type == "ppf":
                norm_xi = normalize_xi_to_gaussian(xi)
            elif self.cfg.designs.norm_type == "log":
                norm_xi = jnp.log(xi)
            else:
                raise ValueError(
                    f"Norm type {self.cfg.designs.norm_type} not recognized."
                )
            log_x = model.sample_from_base(base_sample, norm_theta, norm_xi)
            return jnp.exp(log_x)

        self.sample_likelihood_from_base = sample_likelihood_from_base

        @hk.without_apply_rng
        @hk.transform
        def log_prob_nodrop(x: Array, theta: Array, xi: Array) -> Array:
            """
            Likelihood without dropout (for non-stochastic MCMC use).
            """
            model = make_nsf(
                event_shape=self.EVENT_SHAPE,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=self.z_scale_theta,
                use_resnet=resnet,
                conditional=True,
                activation=self.activation,
                dropout_rate=0.0,  # no dropout
                spline_range_min=self.spline_range_min,
                spline_range_max=self.spline_range_max,
            )
            # Since base is gaussian, transform from lognormal to normal
            log_x = jnp.log(x + 1e-8)
            if self.cfg.designs.norm_type == "ppf":
                norm_xi = normalize_xi_to_gaussian(xi)
            elif self.cfg.designs.norm_type == "log":
                norm_xi = jnp.log(xi)
            else:
                raise ValueError(f"Norm type {self.cfg.designs.norm_type} not recognized. And you better normalize.")
            norm_theta = prior_to_standard_normal(theta)
            lps = model.log_prob(log_x, norm_theta, norm_xi)
            logdetjac = log_trans_logdetjac(x)

            return lps - logdetjac.squeeze()

        self.log_prob_nodrop = log_prob_nodrop

        # Simulator function
        self.simulator = simulate_sir

    def run(self) -> Callable:
        logf, writer = self._init_logging()
        tic = time.time()

        ############### Setting up params to optimize and hyperparams ###############
        # Initialize the nets' params
        prng_seq = hk.PRNGSequence(self.seed)
        design_round = 0

        flow_params = self.log_prob.init(
            next(prng_seq),
            np.zeros((1, *self.EVENT_SHAPE)),
            np.zeros((1, *self.theta_shape)),
            np.zeros((1, *self.xi_shape))
        )

        likelihood_lp_fun = lambda params, prng_key, x, theta, xi: self.log_prob.apply(
                    params, prng_key, x, theta, xi)

        def epig_log_prob_fun(params, y, theta, xi, dropout_key):
            theta_b = theta[jnp.newaxis, :] if theta.ndim == 1 else theta
            xi_b = xi[jnp.newaxis, :] if xi.ndim == 1 else xi
            y_b = y[jnp.newaxis, :] if y.ndim == 1 else y
            lp = likelihood_lp_fun(
                params, dropout_key, y_b, theta_b, xi_b
            )
            return jnp.squeeze(lp)

        def epig_sample_fun(params, theta, xi, sample_keys, dropout_key):
            theta_b = theta[jnp.newaxis, :] if theta.ndim == 1 else theta
            xi_b = xi[jnp.newaxis, :] if xi.ndim == 1 else xi

            if self.epig_antithetic_sampling:
                base_samples = jax.vmap(
                    lambda key: jrandom.normal(key, shape=self.EVENT_SHAPE)
                )(sample_keys)
                paired_base_samples = jnp.stack(
                    (base_samples, -base_samples), axis=1
                ).reshape((2 * sample_keys.shape[0], *self.EVENT_SHAPE))

                def transform_one(base_sample):
                    y = self.sample_likelihood_from_base.apply(
                        params,
                        dropout_key,
                        base_sample[jnp.newaxis, ...],
                        theta_b,
                        xi_b,
                    )
                    return jnp.squeeze(y, axis=0)

                # Reusing dropout_key makes every +/- pair share the same
                # sampled likelihood model; only the Gaussian base latent flips.
                return jax.vmap(transform_one)(paired_base_samples)

            def sample_one(key):
                y = self.sample_likelihood.apply(
                    params,
                    dropout_key,
                    key,
                    1,
                    theta_b,
                    xi_b,
                )
                return jnp.squeeze(y, axis=0)

            return jax.vmap(sample_one)(sample_keys)

        optimizer = optax.chain(optax.clip_by_global_norm(self.grad_clip),
                                optax.adamw(self.learning_rate, b2=self.flow_beta2))
        ema = optax.ema(decay=0.9999, debias=False)
        ema_opt_state = ema.init(flow_params)
        ema_params = flow_params
        if self.xi_optimizer == "Adam":
            optimizer2 = optax.adam(learning_rate=self.schedule, b2=self.xi_beta2)
        else:
            raise ValueError(f"Xi optimizer type {self.xi_optimizer} not recognized.")

        def _sir_total_log_prob(params: hk.Params, x: Array, theta: Array, xi: Array) -> Array:
            if x.shape[1] == 1:
                return self.log_prob_nodrop.apply(params, x, theta, xi)
            conditional_lp = jax.vmap(self.log_prob_nodrop.apply, in_axes=(None, -1, None, -1))(
                params, x[:, jnp.newaxis], theta, xi[:, jnp.newaxis]
            )
            return jnp.sum(conditional_lp, axis=0)

        def _fixed_design_training_data(
            current_x: Array,
            current_theta: Array,
            current_designs: Array,
        ) -> Tuple[Array, Array, Array]:
            if self.static_outputs_sbi is None:
                return current_x, current_theta, current_designs
            return (
                jnp.concatenate([self.static_outputs_sbi[:self.N], current_x], axis=1),
                current_theta,
                jnp.concatenate([self.d[:self.N], current_designs], axis=1),
            )

        def _nle_loss_sir(
            params: hk.Params,
            prng_key: PRNGKey,
            x: Array,
            theta: Array,
            xi: Array,
        ) -> Tuple[Array, Array]:
            x, theta, xi = shuffle_samples(prng_key, x, theta, xi)
            conditional_lps = _sir_total_log_prob(params, x, theta, xi)
            return -jnp.mean(conditional_lps), conditional_lps

        def _fixed_design_eig_sir(
            params: hk.Params,
            prng_key: PRNGKey,
            x: Array,
            theta: Array,
            xi: Array,
        ) -> Tuple[Array, Array]:
            conditional_lp = _sir_total_log_prob(params, x, theta, xi)

            def scan_fun(carry, i):
                contrastive_lps, theta_i = carry
                theta_i = jnp.roll(theta_i, shift=1, axis=0)
                contrastive_lp = _sir_total_log_prob(params, x, theta_i, xi)
                contrastive_lps = jnp.logaddexp(contrastive_lps, contrastive_lp)
                return (contrastive_lps, theta_i), i + 1

            marginal_lp = jax.lax.scan(
                scan_fun, (conditional_lp, theta), jnp.array(range(self.M))
            )[0][0] - jnp.log(self.M + 1)
            eig_terms = conditional_lp - marginal_lp
            finite = jnp.isfinite(eig_terms)
            eig = jnp.sum(jnp.where(finite, eig_terms, 0.0)) / jnp.maximum(jnp.sum(finite), 1)
            return conditional_lp, eig

        def update_fixed_nle_sir(
            params: hk.Params,
            prng_key: PRNGKey,
            opt_state: OptState,
            ema: optax.GradientTransformation,
            ema_opt_state: OptState,
            x: Array,
            theta: Array,
            xi: Array,
        ) -> Tuple[hk.Params, OptState]:
            (loss, conditional_lps), grads = jax.value_and_grad(
                _nle_loss_sir, has_aux=True)(params, prng_key, x, theta, xi)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            ema_params, new_ema_opt_state = ema.update(new_params, ema_opt_state)
            return new_params, new_opt_state, loss, grads, conditional_lps, ema_params, new_ema_opt_state

        def update_fixed_infonce_sir(
            params: hk.Params,
            prng_key: PRNGKey,
            opt_state: OptState,
            ema: optax.GradientTransformation,
            ema_opt_state: OptState,
            final_ys: Array,
            sde_dict: dict,
            theta: Array,
            xi_value: Array,
        ) -> Tuple[hk.Params, OptState]:
            fixed_xi_params = {"xi_mu": xi_value}
            (loss, (conditional_lps, eig, _, x_mean, x_std, d_sim, _)), grads = jax.value_and_grad(
                lf_pce_design_dist_sir, argnums=0, has_aux=True)(
                params,
                fixed_xi_params,
                prng_key,
                final_ys,
                jnp.array(sde_dict['ts'].numpy()),
                theta,
                self.static_outputs_sbi[:self.N] if self.static_outputs_sbi is not None else None,
                self.d[:self.N] if self.d is not None else None,
                likelihood_lp_fun,
                N=self.N,
                M=self.M,
                lam=self.eig_lambda,
                design_min=0.01,
                design_max=100.,
                use_design_dist=False,
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            ema_params, new_ema_opt_state = ema.update(new_params, ema_opt_state)
            return (
                new_params,
                new_opt_state,
                loss,
                grads,
                conditional_lps,
                eig,
                x_mean,
                x_std,
                d_sim,
                ema_params,
                new_ema_opt_state,
            )

        # Initialize designs xi
        flow_params['xi_mu'] = jnp.array(self.xi_mu)
        flow_params['xi_stddev'] = jnp.array(self.xi_stddev)
        xi_params = {key: value for key, value in flow_params.items() if key == 'xi_mu' or key == 'xi_stddev'}

        # Normalize xi values for optimizer
        # Making this range smaller to avoid numerical issues in first round
        design_min = 0.01
        design_max = 100.
        norm_type = self.cfg.designs.norm_type
        scale_factor = 100.

        if norm_type == "inf":
            scale_factor = float(jnp.max(jnp.array([jnp.abs(design_min), jnp.abs(design_max)])))
            xi_params_scaled = {k: jnp.divide(v, scale_factor) for k, v in xi_params.items() if k in ['xi_mu', 'xi_stddev']}
        elif norm_type == "log":
            xi_params_scaled = {k: jnp.log(v) for k, v in xi_params.items() if k in ['xi_mu', 'xi_stddev']}
        elif norm_type == "ppf":
            xi_params_scaled = {k: normalize_xi_to_gaussian(v) for k, v in xi_params.items() if k in ['xi_mu', 'xi_stddev']}
        else:
            raise ValueError(f"Norm type {norm_type} not recognized.")

        flow_params = {key: value for key, value in flow_params.items() if key != 'xi_mu' and key != 'xi_stddev'}

        # Collecting optimal design since overwriting for SIR
        d_hist = []
        best_eig_hist = []
        obs_hist = []
        x_means = []
        median_distances = []

        # LR scheduling items
        sch_init_fn, sch_update_fn = reduce_on_plateau(
            reduce_factor=self.reduce_factor,
            patience=self.patience,
            min_improvement=self.min_improvement,
            cooldown=self.cooldown,
            lr=self.xi_lr_init,
        )
        learning_rate = self.xi_lr_init
        is_baseline = self.design_policy != "optimized"

        # Import "true" SDE data type
        file_path = self.cfg.data.observations
        # Has keys: prior_samples, ys, dt, ts, N, I0, num_samples
        true_sde_dict = torch.load(file_path, map_location="cpu", weights_only=False)
        # Training trajectories are generated online on a 100,000-point grid.
        # Only debug mode needs the historical precomputed prior pool.
        sde_dict = {"ts": torch.linspace(0.0, 100.0, 100000)}
        if self.debug:
            if not self.cfg.data.prior_pool:
                raise ValueError("experiment.debug=true requires data.prior_pool.")
            with open(self.cfg.data.prior_pool, "rb") as file:
                sde_dict = pkl.load(file)

        self.static_outputs_sbi = None
        self.d = None

        # ----- Start SBI-BOED -----
        for design_round in range(self.design_rounds):
            ################# Start Design Optimization #################
            # Initialize the optimizers for the next round of design optimization
            selected_baseline_design = None
            if is_baseline:
                selected_baseline_design = jnp.asarray(select_baseline_design(
                    self.design_policy,
                    design_round,
                    self.seed,
                    design_min,
                    design_max,
                    shape=(1,),
                ))
                xi_params['xi_mu'] = selected_baseline_design
                if norm_type == "inf":
                    xi_params_scaled['xi_mu'] = jnp.divide(selected_baseline_design, scale_factor)
                elif norm_type == "log":
                    xi_params_scaled['xi_mu'] = jnp.log(selected_baseline_design)
                elif norm_type == "ppf":
                    xi_params_scaled['xi_mu'] = normalize_xi_to_gaussian(selected_baseline_design)
                else:
                    raise ValueError(f"Norm type {norm_type} not recognized.")
                opt_state = optimizer.init(flow_params)
            else:
                opt_state = optimizer.init((flow_params, xi_params_scaled))
            ema_opt_state = ema.init(flow_params)
            ema_params = flow_params
            early_best_avg_eig = float("-inf")
            early_bad_steps = 0
            early_best_ema_params = ema_params
            # (Re)set data structs to keep track of best-seen xi_params
            eig_history = deque(maxlen=100)
            xi_mu_history = deque(maxlen=100)
            best_avg_eig = float('-inf')
            best_xi_mu_eig = xi_params['xi_mu']

            # Build the paired theta/SDE pool used throughout this BOED round.
            if design_round == 0:
                if self.debug:
                    boed_sde_pool = jnp.array(sde_dict['final_ys'])
                    boed_theta_pool = jnp.array(sde_dict['theta_0'])
                else:
                    prior_samples, prior_log_probs = sample_lognormal_with_log_probs(
                        next(prng_seq), self.posterior_pool_size
                    )
                    boed_sde_pool, boed_theta_pool, _ = collect_sufficient_sde_samples_prior(
                        self.posterior_pool_size,
                        prior_samples,
                        prior_log_probs,
                        self.device,
                        prng_seq,
                    )
                prior_lp_fun = lambda theta: lognormal_log_prob(theta)
            else:
                boed_theta_pool = posterior_theta_pool
                boed_sde_pool = posterior_sde_pool

            epig_active = self.epig_enabled and not is_baseline
            step_pool_size = self.epig_num_candidates if epig_active else self.N
            if step_pool_size > boed_theta_pool.shape[0]:
                raise ValueError(
                    f"Cannot draw {step_pool_size} paired values from a BOED pool "
                    f"of size {boed_theta_pool.shape[0]}."
                )
            if epig_active and step_pool_size == boed_theta_pool.shape[0]:
                warnings.warn(
                    "epig.num_candidates equals the BOED pool size; every EPIG step "
                    "will reuse the full pool in a different order.",
                    stacklevel=2,
                )

            # Initial design optimization round
            for step in range(self.training_steps):
                tic = time.time()
                theta_indices = jrandom.choice(
                    next(prng_seq),
                    boed_theta_pool.shape[0],
                    shape=(step_pool_size,),
                    replace=False,
                )
                theta_0 = boed_theta_pool[theta_indices]
                final_ys_0 = boed_sde_pool[:, theta_indices]

                if is_baseline:
                    if self.likelihood_objective == "nle":
                        d_sim = jnp.broadcast_to(selected_baseline_design, (self.N, 1))
                        x_step, x_mean, x_std = simulate_sir(
                            d_sim,
                            jnp.array(sde_dict['ts'].numpy()),
                            final_ys_0,
                        )
                        x_train, theta_train, xi_train = _fixed_design_training_data(
                            x_step, theta_0, d_sim)
                        flow_params, opt_state, loss, grads, conditional_lps, ema_params, ema_opt_state = update_fixed_nle_sir(
                            flow_params,
                            next(prng_seq),
                            opt_state,
                            ema,
                            ema_opt_state,
                            x_train,
                            theta_train,
                            xi_train,
                        )
                        _, EIG = _fixed_design_eig_sir(
                            flow_params,
                            next(prng_seq),
                            x_train,
                            theta_train,
                            xi_train,
                        )
                    else:
                        flow_params, opt_state, loss, grads, conditional_lps, EIG, x_mean, x_std, d_sim, ema_params, ema_opt_state = update_fixed_infonce_sir(
                            flow_params,
                            next(prng_seq),
                            opt_state,
                            ema,
                            ema_opt_state,
                            final_ys_0,
                            sde_dict,
                            theta_0,
                            selected_baseline_design,
                        )
                    xi_grads = {"xi_mu": jnp.array(0.0), "xi_stddev": jnp.array(0.0)}
                    xi_updates = {"xi_mu": jnp.array(0.0), "xi_stddev": jnp.array(0.0)}
                    flow_norms = optax.global_norm(grads)
                else:
                    # Optimize the designs using theta_0, flow_params, and xi_params
                    flow_params, xi_params_scaled, opt_state, loss, grads, xi_grads, \
                        xi_updates, EIG, x_mean, x_std, d_sim, flow_norms, conditional_lps, ema_params, ema_opt_state, epig_diagnostics = update_pce(
                            flow_params,
                            xi_params_scaled,
                            next(prng_seq),
                            optimizer,
                            opt_state,
                            ema,
                            ema_opt_state,
                            final_ys_0,
                            sde_dict,
                            likelihood_lp_fun,
                            N=self.N,
                            M=self.M,
                            theta_0=theta_0,
                            lam=self.eig_lambda,
                            opt_round=step,
                            design_min=float(design_min),
                            design_max=design_max,
                            end_sigma=self.end_sigma,
                            importance_sampling=self.importance_sampling,
                            use_design_dist=self.use_design_dist,
                            prev_data=self.static_outputs_sbi[theta_indices] if self.static_outputs_sbi is not None else None,
                            prev_designs=self.d[theta_indices] if self.d is not None else None,
                            epig_log_prob_fun=epig_log_prob_fun,
                            epig_sample_fun=epig_sample_fun,
                            epig_enabled=epig_active,
                            epig_policy=self.epig_policy,
                            epig_K=self.epig_K,
                            epig_S=self.epig_S,
                            epig_temperature=self.epig_temperature,
                            epig_dropout_crn=self.epig_dropout_crn,
                            )

                val_loss = -jnp.mean(conditional_lps)
                should_stop_early = False

                if (not is_baseline) and self.xi_scheduler == "Custom":
                    # Using custom ReduceLROnPlateau scheduler
                    if step == 0:
                        rlrop_state = sch_init_fn(xi_params_scaled)
                    xi_updates, rlrop_state = sch_update_fn(
                        xi_grads,
                        rlrop_state,
                        min_lr=self.xi_lr_end,
                        extra_args={'loss': loss}
                    )
                    # Update learning rate
                    # self.schedule = rlrop_state.lr
                    learning_rate = rlrop_state.lr
                else:
                    learning_rate = self.schedule

                flow_grads = grads[0] if not is_baseline else grads
                if check_for_nans(flow_grads):
                    print("Flow gradients contain NaNs. Resetting to EMA params.")
                    flow_params = ema_params
                    opt_state = optimizer.init(flow_params if is_baseline else (ema_params, xi_params_scaled))
                    ema_opt_state = ema.init(flow_params)
                if (not is_baseline) and check_for_nans(grads[1]):
                    print("Xi gradients contain NaNs. Resetting to EMA params.")
                    # TODO: Make more configureable with the type of normalization chosen
                    xi_params['xi_mu'] = jnp.array(best_xi_mu_eig)
                    if self.cfg.designs.norm_type == "ppf":
                        xi_params_scaled['xi_mu'] = normalize_xi_to_gaussian(xi_params['xi_mu'])
                    elif self.cfg.designs.norm_type == "log":
                        xi_params_scaled['xi_mu'] = jnp.log(xi_params['xi_mu'])
                    else:
                        raise ValueError(f"Norm type {self.cfg.designs.norm_type} not recognized. And you better normalize.")

                    opt_state = optimizer.init((ema_params, xi_params_scaled))
                    ema_opt_state = ema.init(flow_params)

                if is_baseline:
                    xi_params['xi_mu'] = selected_baseline_design
                elif norm_type == "inf":
                    # Setting bounds on the xi_mu values
                    max_bound = jnp.divide(design_max, scale_factor)-0.001
                    xi_params_scaled['xi_mu'] = jnp.clip(
                        xi_params_scaled['xi_mu'],
                        a_min=jnp.divide(design_min, scale_factor),
                        a_max=max_bound
                        )
                    xi_params['xi_mu'] = jnp.multiply(xi_params_scaled['xi_mu'], scale_factor)
                    xi_params['xi_stddev'] = jnp.multiply(xi_params_scaled['xi_stddev'], scale_factor)
                elif norm_type == "log":
                    max_bound = jnp.log(design_max)
                    min_bound = jnp.log(design_min)
                    xi_params_scaled['xi_mu'] = jnp.clip(
                        xi_params_scaled['xi_mu'],
                        a_min=min_bound,
                        a_max=max_bound
                        )
                    xi_params['xi_mu'] = jnp.exp(
                        xi_params_scaled['xi_mu'])
                    xi_params['xi_stddev'] = jnp.exp(
                        xi_params_scaled['xi_stddev'])
                elif norm_type == "ppf":
                    max_bound = normalize_xi_to_gaussian(design_max)
                    min_bound = normalize_xi_to_gaussian(design_min)
                    xi_params_scaled['xi_mu'] = jnp.clip(
                        xi_params_scaled['xi_mu'],
                        a_min=min_bound,
                        a_max=max_bound
                        )
                    xi_params['xi_mu'] = inverse_normalize_xi(
                        xi_params_scaled['xi_mu'])
                    xi_params['xi_stddev'] = inverse_normalize_xi(
                        xi_params_scaled['xi_stddev'])
                else:
                    raise ValueError(f"Norm type {norm_type} not recognized.")

                # calculate the rolling average
                eig_history.append(EIG)
                xi_mu_history.append(xi_params['xi_mu'])
                rolling_average_eig = jnp.mean(np.array(eig_history))
                if rolling_average_eig > best_avg_eig:
                    best_avg_eig = rolling_average_eig
                    best_xi_mu_eig = jnp.mean(np.array(xi_mu_history))

                if is_baseline and self.baseline_early_stopping:
                    best_avg_eig_float = float(jax.device_get(best_avg_eig))
                    if best_avg_eig_float > early_best_avg_eig + self.baseline_early_stopping_scale:
                        early_best_avg_eig = best_avg_eig_float
                        early_bad_steps = 0
                        early_best_ema_params = ema_params
                    else:
                        early_bad_steps += 1

                    should_stop_early = early_bad_steps >= self.baseline_early_stopping_patience

                run_time = time.time()-tic
                xi_mu_value = float(jnp.squeeze(xi_params['xi_mu']))
                xi_std_value = float(jnp.squeeze(xi_params['xi_stddev']))
                xi_update_value = float(jnp.squeeze(xi_updates['xi_mu']))
                early_bad_steps_log = early_bad_steps if is_baseline else ""
                early_best_avg_eig_log = (
                    early_best_avg_eig
                    if is_baseline and np.isfinite(early_best_avg_eig)
                    else ""
                )
                print(f"STEP: {step:5d}; Xi Mu: {xi_mu_value:.4f}; Xi Stddev: {xi_std_value:.4f}; Xi mu Updates: {xi_update_value:.4e}; Loss: {loss:.4f}; EIG: {EIG:.4f}; Run time: {run_time:.4f}, Flow Grad Norm: {flow_norms:.4f}, Val Loss: {val_loss:.4f}")
                if epig_active:
                    print(
                        f"EPIG: policy={self.epig_policy}; "
                        f"dropout_crn={self.epig_dropout_crn}; "
                        f"score mean={epig_diagnostics.score_mean:.4f}; "
                        f"selected mean={epig_diagnostics.selected_score_mean:.4f}; "
                        f"valid={epig_diagnostics.valid_score_fraction:.3f}; "
                        f"selected valid={epig_diagnostics.selected_valid_fraction:.3f}; "
                        f"fallback={bool(epig_diagnostics.fallback_used)}; "
                        f"finite(sample/lp/ratio)="
                        f"{epig_diagnostics.sample_finite_fraction:.3f}/"
                        f"{epig_diagnostics.log_prob_finite_fraction:.3f}/"
                        f"{epig_diagnostics.ratio_finite_fraction:.3f}; "
                        f"max|sample/lp/centered-lp/ratio|="
                        f"{epig_diagnostics.sample_abs_max:.3e}/"
                        f"{epig_diagnostics.log_prob_abs_max:.3e}/"
                        f"{epig_diagnostics.centered_log_prob_abs_max:.3e}/"
                        f"{epig_diagnostics.ratio_abs_max:.3e}"
                    )

                # Log the results
                writer.writerow({
                    'STEP': step,
                    'Xi_mu': xi_params['xi_mu'],
                    'Xi_stddev': xi_params['xi_stddev'],
                    'Loss': loss,
                    'EIG': EIG,
                    'time':float(run_time),
                    'seed': self.seed,
                    'lambda': self.eig_lambda,
                    'design_round': design_round,
                    'design_policy': self.design_policy,
                    'likelihood_objective': self.likelihood_objective,
                    'early_stopping_bad_steps': early_bad_steps_log,
                    'early_stopping_best_avg_eig': early_best_avg_eig_log,
                })
                logf.flush()

                if self.cfg.wandb.use_wandb:
                    step_metric_name = f"boed_{design_round}/step"
                    wandb.define_metric(step_metric_name)
                    wandb.define_metric(f"boed_{design_round}/*", step_metric=step_metric_name)
                    wandb_payload = {
                        f"boed_{design_round}/loss": loss,
                        f"boed_{design_round}/design_mu": xi_params['xi_mu'],
                        f"boed_{design_round}/design_stddev": xi_params['xi_stddev'],
                        f"boed_{design_round}/xi_mu_grads": xi_grads['xi_mu'],
                        f"boed_{design_round}/best_xi_mu_eig": best_xi_mu_eig,
                        f"boed_{design_round}/best_avg_eig": best_avg_eig,
                        f"boed_{design_round}/EIG": EIG,
                        f"boed_{design_round}/mean_scaled_x": x_mean,
                        f"boed_{design_round}/std_scaled_x": x_std,
                        f"boed_{design_round}/learning_rate": learning_rate,
                        f"boed_{design_round}/flow grad norms": flow_norms,
                        f"boed_{design_round}/val_loss": val_loss,
                        f"boed_{design_round}/design_policy": self.design_policy,
                        f"boed_{design_round}/likelihood_objective": self.likelihood_objective,
                        step_metric_name: step,
                        }
                    if is_baseline:
                        wandb_payload.update({
                            f"boed_{design_round}/early_stopping_bad_steps": early_bad_steps,
                            f"boed_{design_round}/early_stopping_best_avg_eig": early_best_avg_eig,
                        })
                    if epig_active:
                        wandb_payload.update({
                            f"boed_{design_round}/epig/policy": self.epig_policy,
                            f"boed_{design_round}/epig/score_mean": epig_diagnostics.score_mean,
                            f"boed_{design_round}/epig/score_max": epig_diagnostics.score_max,
                            f"boed_{design_round}/epig/selected_score_mean": epig_diagnostics.selected_score_mean,
                            f"boed_{design_round}/epig/effective_sample_size": epig_diagnostics.effective_sample_size,
                            f"boed_{design_round}/epig/max_importance_weight": epig_diagnostics.max_importance_weight,
                            f"boed_{design_round}/epig/valid_score_fraction": epig_diagnostics.valid_score_fraction,
                            f"boed_{design_round}/epig/selected_valid_fraction": epig_diagnostics.selected_valid_fraction,
                            f"boed_{design_round}/epig/fallback_used": epig_diagnostics.fallback_used,
                            f"boed_{design_round}/epig/sample_finite_fraction": epig_diagnostics.sample_finite_fraction,
                            f"boed_{design_round}/epig/log_prob_finite_fraction": epig_diagnostics.log_prob_finite_fraction,
                            f"boed_{design_round}/epig/ratio_finite_fraction": epig_diagnostics.ratio_finite_fraction,
                            f"boed_{design_round}/epig/sample_abs_max": epig_diagnostics.sample_abs_max,
                            f"boed_{design_round}/epig/log_prob_abs_max": epig_diagnostics.log_prob_abs_max,
                            f"boed_{design_round}/epig/centered_log_prob_abs_max": epig_diagnostics.centered_log_prob_abs_max,
                            f"boed_{design_round}/epig/ratio_abs_max": epig_diagnostics.ratio_abs_max,
                        })
                    wandb.log(wandb_payload)

                if should_stop_early:
                    print(
                        f"Early stopping baseline likelihood training at step {step}; "
                        f"best avg EIG {early_best_avg_eig:.6f}"
                    )
                    ema_params = early_best_ema_params
                    flow_params = early_best_ema_params
                    break

            ############# Finished design optimization & reset to best checkpointed params #############
            xi_params['xi_mu'] = jnp.array(best_xi_mu_eig)
            if self.cfg.designs.norm_type == "inf":
                xi_params_scaled['xi_mu'] = jnp.divide(xi_params['xi_mu'], scale_factor)
            elif self.cfg.designs.norm_type == "ppf":
                xi_params_scaled['xi_mu'] = normalize_xi_to_gaussian(xi_params['xi_mu'])
            elif self.cfg.designs.norm_type == "log":
                xi_params_scaled['xi_mu'] = jnp.log(xi_params['xi_mu'])
            else:
                raise ValueError(f"Norm type {self.cfg.designs.norm_type} not recognized. And you better normalize.")

            flow_params = ema_params

            ############# Refine likelihood using SBC #############
            # Can either update with the MI-based or KL-based optimization... use KL for now but know you can reuse
            # Need to still generate prior & likelihood samples for SBI
            if design_round == 0:
                thetas_sbi, thetas_sbi_lp = sample_lognormal_with_log_probs(
                    next(prng_seq), self.posterior_pool_size
                )
            else:
                thetas_sbi, thetas_sbi_lp = run_mcmc(
                    next(prng_seq),
                    mcmc_posterior,
                    theta_0,
                    self.num_adapt_steps,
                    self.posterior_pool_size,
                )
            final_ys_sbi, thetas_sbi, thetas_sbi_lp = collect_sufficient_sde_samples_prior(
                self.posterior_pool_size,
                thetas_sbi,
                thetas_sbi_lp,
                self.device,
                prng_seq,)

            if self.d is None:
                x_sbi, _, _ = simulate_sir(
                    jnp.broadcast_to(
                        xi_params['xi_mu'], (self.posterior_pool_size, 1)),
                    jnp.array(sde_dict['ts'].numpy()),
                    final_ys_sbi)
            else:
                # TODO: Make sure that the outputs from this correspond with how likelihood was trained
                vectorized_simulator = jax.vmap(
                    self.simulator, in_axes=(1, None, None))
                x_sbi, _, _ = vectorized_simulator(
                    jnp.concatenate((
                        self.d,
                        jnp.broadcast_to(
                            xi_params['xi_mu'], (self.posterior_pool_size, 1)
                        ),
                        ), axis=1),
                    jnp.array(sde_dict['ts'].numpy()),
                    final_ys_sbi
                    )
                x_sbi = x_sbi.squeeze().T

            # BUG: The worsening likleihood might be bc you're training with dropout for KL refinement
            @jax.jit
            def generalized_log_prob_fun(params, x, theta, xi):
                if x.shape[1] == 1:
                    conditional_lp = self.log_prob_nodrop.apply(params, x, theta, xi)
                else:
                    # prng_keys = jax.random.split(prng_key, num=x.shape[1])
                    conditional_lp = jax.vmap(self.log_prob_nodrop.apply, in_axes=(None, -1, None, -1))(
                        params, x[:, jnp.newaxis], theta, xi[:, jnp.newaxis]
                    )
                    conditional_lp = jnp.sum(conditional_lp, axis=0)
                return conditional_lp

            if self.d is None:
                sbc_prior_sample_fun = lambda samples: sample_lognormal_with_log_probs(next(prng_seq), samples)
            else:
                sbc_prior_sample_fun = lambda samples: run_mcmc(
                    next(prng_seq), prior_lp, loglikelihood, mcmc_posterior, theta_0, self.num_adapt_steps, samples)

            # TODO: Maybe customize this optmiizer to be unique
            # TODO: add in wandb logging of the validation loss
            optimizer = optax.chain(optax.clip_by_global_norm(self.grad_clip),
                                    optax.adamw(self.learning_rate, b2=self.flow_beta2))
            opt_state = optimizer.init(flow_params)
            ema_refine = optax.ema(decay=0.9999, debias=False)
            ema_refine_opt_state = ema_refine.init(flow_params)
            ema_refine_params = flow_params
            best_kl_loss = float('inf')

            if self.d is None:
                sbi_d = jnp.broadcast_to(
                    xi_params['xi_mu'], (self.posterior_pool_size, 1)
                )
            else:
                sbi_d = jnp.concatenate(
                    (
                        self.d,
                        jnp.broadcast_to(
                            xi_params['xi_mu'], (self.posterior_pool_size, 1)
                        ),
                    ),
                    axis=1,
                )

            x_sbi, thetas_sbi, sbi_d, x_sbi_val, thetas_sbi_val, sbi_d_val = split_data_for_validation_jax(
                x_sbi,
                thetas_sbi,
                sbi_d,
                next(prng_seq),
                validation_fraction=0.1
            )

            key, sub_key = jrandom.split(next(prng_seq))

            for step in range(self.sbi_train_steps):
                key, sub_key = jrandom.split(key)
                x_sbi, thetas_sbi, sbi_d = shuffle_samples(
                    next(prng_seq), x_sbi, thetas_sbi, sbi_d)

                if self.sbc_lambda == 0:
                    kl_loss, grads = jax.value_and_grad(kl_loss_fn_general)(
                        flow_params,
                        # sub_key,
                        x_sbi,
                        thetas_sbi,
                        sbi_d,
                        generalized_log_prob_fun
                        )
                else:
                    kl_loss, grads = jax.value_and_grad(kl_sbc_loss_fn_general)(
                        flow_params,
                        x_sbi,
                        thetas_sbi,
                        sbc_prior_sample_fun,
                        generalized_log_prob_fun,
                        self.sbc_samples,
                        self.sbc_lambda,
                        xi=sbi_d
                        )
                updates, opt_state = optimizer.update(grads, opt_state, flow_params)
                flow_params = optax.apply_updates(flow_params, updates)
                ema_refine_params, ema_refine_opt_state = ema_refine.update(flow_params, ema_refine_opt_state)
                key, sub_key = jrandom.split(key)

                val_loss = generalized_log_prob_fun(flow_params, x_sbi_val, thetas_sbi_val, sbi_d_val)

                if kl_loss < best_kl_loss:
                    best_kl_loss = kl_loss
                print(f"KL Loss: {kl_loss:.4f}")
                print(f"val KL Loss: {-jnp.mean(val_loss):.4f}")
                if self.cfg.wandb.use_wandb:
                    # TODO: see if this fixes the issue of not logging the refine KL loss as a line plot
                    step_metric_name = f"boed_{design_round}/refine_step"
                    wandb.define_metric(step_metric_name)
                    wandb.define_metric(f"boed_{design_round}/*", step_metric=step_metric_name)
                    wandb.log({
                        f"boed_{design_round}/kl_loss": kl_loss,
                        f"boed_{design_round}/kl_val_loss": -jnp.mean(val_loss),
                        step_metric_name: step,
                        })
            flow_params = ema_refine_params


            ############# Log experiment #############
            # Use best xi_mu corresponding to best EIG for SBI
            best_eig_hist.append(best_avg_eig)
            self.d_sim = xi_params['xi_mu']

            # Simulate observed value
            x_obs, _, _ = self.simulator(
                xi_params['xi_mu'],
                jnp.array(true_sde_dict['ts'].numpy()),
                jnp.array(true_sde_dict['ys'].numpy())
                )

            # Set static outputs for SBI
            if self.static_outputs_sbi is None:
                self.static_outputs_sbi = jnp.broadcast_to(
                    x_obs, (self.posterior_pool_size, 1)
                )
                self.d = jnp.broadcast_to(
                    xi_params['xi_mu'], (self.posterior_pool_size, 1)
                )
            else:
                self.static_outputs_sbi = jnp.concatenate((self.static_outputs_sbi,
                                                            jnp.broadcast_to(x_obs, (self.posterior_pool_size, 1))), axis=1)
                self.d = jnp.concatenate((self.d,
                                          jnp.broadcast_to(xi_params['xi_mu'], (self.posterior_pool_size, 1))), axis=1)


            ############# Record LC2ST Metrics #############
            # Log posterior samples from mcmc and other metrics to wandb
            @jax.jit
            def standard_normal_to_prior(z):
                theta_loc = jnp.log(jnp.array([0.5, 0.1]))
                theta_covmat = jnp.eye(2) * 0.5 ** 2  # Covariance matrix
                std_devs = jnp.sqrt(jnp.diag(theta_covmat))  # Standard deviations [0.5, 0.5]
                # Inverse transformation
                log_theta = z * std_devs + theta_loc
                theta = jnp.exp(log_theta)
                return theta


            @jax.jit
            def prior_lp_logdetjac(x):
                def jacobian_fn(xi):
                    return jax.jacfwd(standard_normal_to_prior)(xi)
                jacobians = jax.vmap(jacobian_fn)(x)
                logdetjac = jax.vmap(jnp.linalg.slogdet)(jacobians)[1]
                return logdetjac.reshape(-1, 1)


            if design_round == 0:
                # Shape of x_obs is [1,1]
                prior_lp = lambda theta: prior_lp_fun(standard_normal_to_prior(theta)).squeeze() + \
                      prior_lp_logdetjac(theta).squeeze()
                mcmc_posterior = lambda theta: self.log_prob_nodrop.apply(
                    flow_params,
                    x_obs,
                    standard_normal_to_prior(theta),
                    jnp.array([[xi_params['xi_mu']]])
                    ).squeeze() + prior_lp(theta)
                loglikelihood = lambda theta: self.log_prob_nodrop.apply(
                    flow_params,
                    x_obs,
                    standard_normal_to_prior(theta)[None,:],
                    jnp.array([[xi_params['xi_mu']]])
                    ).squeeze()
            else:
                mcmc_posterior = lambda theta: jnp.sum(jax.vmap(
                    self.log_prob_nodrop.apply, in_axes=(None, -1, None, -1))(
                        flow_params,
                        self.static_outputs_sbi[0,:][None,None,:],
                        standard_normal_to_prior(theta),
                        self.d[0,:][None,None,:]
                    )).squeeze() + prior_lp(theta)

                loglikelihood = lambda theta: jnp.sum(jax.vmap(
                    self.log_prob_nodrop.apply, in_axes=(None, -1, None, -1))(
                        flow_params,
                        self.static_outputs_sbi[0,:][None,None,:],
                        standard_normal_to_prior(theta)[None,:],
                        self.d[0,:][None,None,:]
                    )).squeeze()

            posterior_theta_pool, posterior_log_prob_pool = run_mcmc(
                next(prng_seq),
                mcmc_posterior,
                theta_0,
                self.num_adapt_steps,
                self.posterior_pool_size,
            )

            # Paired posterior theta/SDE pool for diagnostics and the next BOED round.
            posterior_sde_pool, posterior_theta_pool, posterior_log_prob_pool = collect_sufficient_sde_samples_prior(
                self.posterior_pool_size,
                posterior_theta_pool,
                posterior_log_prob_pool,
                self.device,
                prng_seq,
            )

            # Need to simulate posterior predictive points for LC2ST
            if self.d is None:
                xs, _, _ = simulate_sir(
                    jnp.broadcast_to(
                        xi_params['xi_mu'], (self.posterior_pool_size, 1)),
                    jnp.array(sde_dict['ts'].numpy()),
                    final_ys_sbi)
            else:
                # TODO: Make sure that the outputs from this correspond with how likelihood was trained
                vectorized_simulator = jax.vmap(
                    self.simulator, in_axes=(1, None, None))
                xs, _, _ = vectorized_simulator(
                    self.d,
                    jnp.array(sde_dict['ts'].numpy()),
                    final_ys_sbi
                    )
                xs = xs.squeeze().T

            # LC2ST uses scikit-learn cross-validation, which requires CPU arrays.
            x_o = torch.from_numpy(
                np.array(self.static_outputs_sbi[0, :][None, :], copy=True)
            ).float()
            posterior_theta_pool_torch = torch.from_numpy(
                np.array(posterior_theta_pool, copy=True)
            ).float()
            xs = torch.from_numpy(np.array(xs, copy=True)).float()
            thetas = torch.from_numpy(np.array(theta_0, copy=True)).float()

            if design_round == 0: xs = xs.unsqueeze(1)
            lc2st = LC2ST(
                thetas=thetas,
                xs=xs[:thetas.shape[0]],
                posterior_samples=posterior_theta_pool_torch[:thetas.shape[0]],
                seed=self.seed,
                num_folds=3,
                num_ensemble=4,
                classifier="mlp",
                z_score=True,
                num_trials_null=100,
                permutation=True,
            )

            lc2st.train_under_null_hypothesis()
            lc2st.train_on_observed_data()

            theta_o = posterior_theta_pool_torch
            statistic = lc2st.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_o)
            print("L-C2ST statistic on observed data:", statistic)
            p_value = lc2st.p_value(theta_o=theta_o, x_o=x_o)
            print("P-value for L-C2ST:", p_value)

            # Decide whether to reject the null hypothesis at a significance level alpha
            alpha = 0.05  # 95% confidence level
            reject = lc2st.reject_test(theta_o=theta_o, x_o=x_o, alpha=alpha)
            print(f"Reject null hypothesis at alpha = {alpha}:", reject)

            ############ Save params/data & draw posterior sample that becomes new theta_0 ###############
            # Save the likelihood here to debug posterior training
            if self.cfg.experiment.save_params:
                flow_save_key = f"design_round_{design_round}_flow_params"
                objects = {flow_save_key: jax.device_get(flow_params),
                           "theta_0": jax.device_get(theta_0),
                           "best_xi_mu_eig": jax.device_get(best_xi_mu_eig),
                           "final_ys": jax.device_get(final_ys_0),
                           "sde_ts": jax.device_get(sde_dict['ts']),
                           "design_policy": self.design_policy,
                           "likelihood_objective": self.likelihood_objective,
                           'LC2ST_statistic': statistic,
                           'LC2ST_p_value': p_value,
                           'LC2ST_reject': reject,}
                with open(f"{self.subdir}/{flow_save_key}.pkl", "wb") as f:
                    pkl.dump(objects, f)

            # Observe the x_post value using the surrogate simulator
            if design_round == 0:
                x_post, _, _ = self.simulator(self.d[-self.posterior_pool_size:],
                                              jnp.array(sde_dict['ts'].numpy()),
                                              posterior_sde_pool)
            else:
                # TODO: double check the outputs from this correspond with how likelihood was trained
                vectorized_simulator = jax.vmap(self.simulator, in_axes=(1, None, None))
                x_post, _, _ = vectorized_simulator(
                    self.d[-self.posterior_pool_size:],
                    jnp.array(sde_dict['ts'].numpy()),
                    posterior_sde_pool
                )
                x_post = x_post.squeeze().T

            median_distance = jnp.median(jnp.linalg.norm(self.static_outputs_sbi - x_post, ord=2, axis=1))
            print(f"Design round {design_round} median distance: {median_distance}")

            if self.cfg.wandb.use_wandb:
                true_x = jnp.array([0.7399])
                true_y = jnp.array([0.0924])
                plt.hist2d(
                    posterior_theta_pool[:, 0],
                    posterior_theta_pool[:, 1],
                    range=[[0.2, 1.6], [0., 0.5]],
                    bins=100,
                )
                plt.scatter(true_x, true_y, color='red')
                # Get the original current working directory
                original_cwd = hydra.utils.get_original_cwd()
                plot_directory = os.path.join(original_cwd, 'temp_plot_data')
                os.makedirs(plot_directory, exist_ok=True)
                plot_path = os.path.join(plot_directory, 'post_histogram.png')
                plt.savefig(plot_path)
                plt.close()
                wandb.log({f'Design round {design_round} posteriors': wandb.Image(plot_path),
                           f"boed_{design_round}/variance": jnp.var(posterior_theta_pool),
                           f"boed_{design_round}/median_distance": median_distance,
                           f"boed_{design_round}/LC2ST_statistic": statistic,
                           f"boed_{design_round}/LC2ST_p_value": p_value,
                           f"boed_{design_round}/LC2ST_reject": reject,
                           })


            # Reinitialize xi with prev_xi in mind
            design_min = xi_params['xi_mu']
            self.xi = (100. - xi_params['xi_mu']) / 2. + xi_params['xi_mu']
            self.d_sim = self.xi

            x_means.append(x_mean)
            obs_hist.append(x_obs)
            d_hist.append(xi_params['xi_mu'])
            median_distances.append(median_distance)
            xi_params['xi_mu'] = self.xi
            xi_params['xi_stddev'] = self.xi_stddev

            # Reset the xi_params_scaled for optimization
            if norm_type == "inf":
                xi_params_scaled['xi_mu'] = jnp.divide(self.xi, scale_factor)
                xi_params_scaled['xi_stddev'] = jnp.divide(self.xi_stddev, scale_factor)
            elif norm_type == "log":
                xi_params_scaled = {k: jnp.log(v) for k, v in xi_params.items() if k in ['xi_mu', 'xi_stddev']}
            elif norm_type == "ppf":
                xi_params_scaled = {k: normalize_xi_to_gaussian(v) for k, v in xi_params.items() if k in ['xi_mu', 'xi_stddev']}
            else:
                raise ValueError(f"Norm type {norm_type} not recognized.")

            # optionally reset the params
            if self.cfg.flow_params.reset_flow and design_round < self.design_rounds - 1:
                del flow_params
                flow_params = self.log_prob.init(
                    next(prng_seq),
                    np.zeros((1, *self.EVENT_SHAPE)),
                    np.zeros((1, *self.theta_shape)),
                    np.zeros((1, *self.xi_shape))
                )


        if self.cfg.wandb.use_wandb:
            wandb.log({f"boed_{design_round}/final_design_EIG": jnp.sum(np.array(best_eig_hist))})

        print(f"final design EIG: {jnp.sum(np.array(best_eig_hist))}")

        if self.cfg.experiment.save_params:
            objects = {
                'x_means': jax.device_get(x_means),
                'x_obs': jax.device_get(obs_hist),
                'd_hist': jax.device_get(d_hist),
                'best_eig_hist': jax.device_get(best_eig_hist),
                'post_samples': jax.device_get(posterior_theta_pool),
                'post_log_probs': jax.device_get(posterior_log_prob_pool),
                'median_distances': jax.device_get(median_distances),
                'design_policy': self.design_policy,
                'likelihood_objective': self.likelihood_objective,
            }
            with open(f"{self.subdir}/{self.cfg.experiment.save_name}.pkl", "wb") as f:
                pkl.dump(objects, f)


    def _init_logging(self):
        path = os.path.join(self.subdir, 'log.csv')
        logf = open(path, 'a')
        fieldnames = [
            'STEP',
            'Xi_mu',
            'Xi_stddev',
            'Loss',
            'EIG',
            'time',
            'seed',
            'lambda',
            'design_round',
            'design_policy',
            'likelihood_objective',
            'early_stopping_bad_steps',
            'early_stopping_best_avg_eig',
        ]
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat(path).st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer



@hydra.main(version_base=None, config_path=".", config_name="config_sir")
def main(cfg):
    fname = os.getcwd() + '/latest.pt'
    if os.path.exists(fname):
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = pkl.load(f)
        print(f"STEP: {workspace.step:5d}; Xi: {workspace.xi};\
             Xi Grads: {workspace.xi_grads}; Loss: {workspace.loss}")
    else:
        workspace = Workspace(cfg)

    workspace.run()


if __name__ == "__main__":
    main()
