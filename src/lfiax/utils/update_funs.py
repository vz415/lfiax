from functools import partial
import matplotlib.pyplot as plt
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import haiku as hk
import optax

from lfiax.utils.oed_losses import lf_pce_design_dist_sir, lf_ace_design_dist_sir

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
def inverse_normalize_xi(normalized_x):
    # Inverse of the probit function (normal CDF)
    scaled_x = jax.scipy.stats.norm.cdf(normalized_x)
    # Rescale back to original range
    x = scaled_x * 100.0
    return x


@jax.jit
def normalize_xi_to_gaussian(x):
    scaled_x = x / 100.0
    # Apply probit function (inverse normal CDF)
    normalized_x = jax.scipy.stats.norm.ppf(scaled_x)
    return normalized_x


def compute_average_norm(grads):
    norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
    flat_norms = jax.tree_util.tree_leaves(norms)
    average_norm = jnp.mean(jnp.array(flat_norms))
    return average_norm

def plot_eig_histogram(d_sim, EIGs, plot_directory, n_bins=20):
    # Create bins for the design space
    bins = np.linspace(d_sim.min(), d_sim.max(), n_bins+1)

    # Digitize the d_sim values into bins
    bin_indices = np.digitize(d_sim, bins) - 1

    # Calculate the total EIG for each bin
    eig_per_bin = np.zeros(n_bins)
    for i in range(n_bins):
        eig_per_bin[i] = EIGs[bin_indices == i].sum()

    # Normalize the EIG values
    eig_per_bin_normalized = eig_per_bin / eig_per_bin.sum()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(bins[:-1], eig_per_bin_normalized, width=np.diff(bins), align="edge", alpha=0.7)
    plt.xlabel('Design (d_sim)')
    plt.ylabel('Normalized EIG')
    plt.title('Distribution of EIGs over Design Space')

    # Add value labels on top of each bar
    for i, v in enumerate(eig_per_bin_normalized):
        plt.text(bins[i], v, f'{v:.2f}', ha='left', va='bottom', rotation=90)

    # Save plot
    plot_path = os.path.join(plot_directory, 'eig_distribution_histogram.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {plot_path}")


@partial(jax.jit, static_argnums=[6,7,8,9,10,11,12,14,15,16,17])
def compute_grads_and_loss_sir_ace(
    flow_params: hk.Params,
    post_params: hk.Params,
    xi_params: hk.Params,
    prng_key: PRNGKey,
    final_ys: Array,
    sde_dict_ts: Array,
    likelihood_lp: Callable,
    prior_lp_fun: Callable,
    prior_sample_fun: Callable,
    post_lp_fun: Callable,
    post_sample_fun: Callable,
    N: int,
    M: int,
    theta_0: Array,
    lam: float,
    sbc_samples: int,
    sbc_lambda: float,
    design_min: float = 0.01,
    design_max: float = 100.,
    ):
    '''Basic compute grads of InfoNCE objective.'''
    (loss, (conditional_lp, EIG, EIGs, x_mean, x_std, d_sim, prior_post_diff, prior_post_cont)), grads = jax.value_and_grad(
        lf_ace_design_dist_sir, argnums=[0,1,2], has_aux=True)(
        flow_params, post_params, xi_params, prng_key, final_ys, sde_dict_ts,
        theta_0, likelihood_lp, prior_lp_fun, prior_sample_fun, post_lp_fun, post_sample_fun,
        N=N, M=M, lam=lam, design_min=design_min, design_max=design_max,
        sbc_samples=sbc_samples, sbc_lambda=sbc_lambda
        )

    return grads, loss, conditional_lp, EIG, x_mean, x_std, d_sim, EIGs, prior_post_diff, prior_post_cont

# @partial(jax.jit, static_argnums=[7,8,9,11,12,13,14])
@partial(
    jax.jit,
    static_argnums=[5, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22],
)
def compute_grads_and_loss_sir_pce(
    flow_params: hk.Params,
    xi_params: hk.Params,
    prng_key: PRNGKey,
    final_ys: Array,
    sde_dict_ts: Array,
    likelihood_lp: Callable,
    N: int,
    M: int,
    theta_0: Array,
    lam: float,
    prev_data: Optional[Array] = None,
    prev_designs: Optional[Array] = None,
    design_min: float = 0.01,
    design_max: float = 100.,
    use_design_dist: bool = True,
    epig_log_prob_fun: Optional[Callable] = None,
    epig_sample_fun: Optional[Callable] = None,
    epig_enabled: bool = False,
    epig_policy: str = "top_k",
    epig_K: int = 8,
    epig_S: int = 1,
    epig_temperature: float = 1.0,
    epig_dropout_crn: bool = False,
    ):
    '''Basic compute grads of InfoNCE objective.'''
    (loss, (conditional_lp, EIG, EIGs, x_mean, x_std, d_sim, epig_diagnostics)), grads = jax.value_and_grad(
        lf_pce_design_dist_sir, argnums=[0,1], has_aux=True)(
        flow_params,
        xi_params,
        prng_key,
        final_ys,
        sde_dict_ts,
        theta_0,
        prev_data=prev_data,
        prev_designs=prev_designs,
        log_prob_fun=likelihood_lp,
        N=N,
        M=M,
        lam=lam,
        design_min=design_min,
        design_max=design_max,
        use_design_dist=use_design_dist,
        epig_log_prob_fun=epig_log_prob_fun,
        epig_sample_fun=epig_sample_fun,
        epig_enabled=epig_enabled,
        epig_policy=epig_policy,
        epig_K=epig_K,
        epig_S=epig_S,
        epig_temperature=epig_temperature,
        epig_dropout_crn=epig_dropout_crn,
        )

    return grads, loss, conditional_lp, EIG, x_mean, x_std, d_sim, EIGs, epig_diagnostics

def update_ace(
    flow_params: hk.Params,
    post_params: hk.Params,
    xi_params: hk.Params, # Note: these are passed in scaled
    prng_key: PRNGKey,
    opt_state: OptState,
    optimizer: optax.GradientTransformation,
    # opt_state_post: OptState,
    opt_state_xi: OptState,
    final_ys: Array,
    sde_dict: dict,
    likelihood_lp: Callable,
    prior_lp_fun: Callable,
    prior_sample_fun: Callable,
    post_lp_fun: Callable,
    post_sample_fun: Callable,
    N: int,
    M: int,
    theta_0: Array,
    lam: float,
    opt_round: int,
    sbc_samples: int,
    sbc_lambda: float,
    design_min: float = 0.01,
    design_max: float = 100.,
    xi_stddev: float = 0.3,
    end_sigma: float = 0.01,
    importance_sampling: bool = False
) -> Tuple[hk.Params, OptState]:
    """Single SGD update step for design optimization."""
    grads, loss, _, EIG, x_mean, x_std, d_sim, EIGs, prior_post_diff, prior_post_cont = compute_grads_and_loss_sir_ace(
        flow_params, post_params, xi_params, prng_key, final_ys,
        jnp.array(sde_dict['ts'].numpy()), likelihood_lp, prior_lp_fun, prior_sample_fun,
        post_lp_fun, post_sample_fun, N, M, theta_0, lam, sbc_samples, sbc_lambda, design_min, design_max)

    # TODO: Add the posterior update steps
    combo_params = (flow_params, post_params, xi_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, combo_params)
    new_params, new_post_params, new_xi_params = optax.apply_updates(combo_params, updates)

    # Calculate grads of flow and post params to plot for debugging
    flow_norms = compute_average_norm(grads[0])
    post_norms = compute_average_norm(grads[1])

    # Exponential schedule for standard deviation
    decay_rate = 10.
    decay_constant = 10_000 / decay_rate
    start_sigma = xi_stddev
    new_xi_params['xi_stddev'] = end_sigma/100. + (start_sigma - end_sigma) * jnp.exp(-opt_round / decay_constant)/100.

    if importance_sampling:
        # Importance sampling step for designs
        # d_sim: shape [N,1]
        # EIGs: shape [N,]
        pdf_values = jax.scipy.stats.norm.pdf(
            d_sim, loc=xi_params['xi_mu']*100., scale=xi_params['xi_stddev']*100.)
        truncation_correction = jax.scipy.stats.norm.cdf(100., loc=xi_params['xi_mu']*100., scale=xi_params['xi_stddev']*100.) - \
            jax.scipy.stats.norm.cdf(0., loc=xi_params['xi_mu']*100., scale=xi_params['xi_stddev']*100.)
        adjusted_pdf_values = pdf_values / truncation_correction
        design_log_probs = jnp.log(adjusted_pdf_values)
        EIGs_shifted = jnp.log(EIGs - jnp.min(EIGs) + 1e-6)
        EIGs_max_shifted = EIGs_shifted - jnp.max(EIGs_shifted)
        combo_log_probs = design_log_probs.squeeze() + EIGs_max_shifted
        post_design_log_probs = jnp.exp(combo_log_probs) / jnp.sum(jnp.exp(combo_log_probs))
        post_design_samples = jrandom.choice(prng_key, d_sim, shape=(len(d_sim),), replace=True, p=post_design_log_probs)
        post_design_mean = jnp.mean(post_design_samples)

        # Average the importance sampled mean and the gradient-based mean
        new_xi_params['xi_mu'] = jnp.mean(jnp.array([post_design_mean, new_xi_params['xi_mu']*100.]))/100.
    xi_grads = grads[2]
    xi_updates = updates[2]
    return new_params, new_post_params, new_xi_params, new_opt_state, loss, grads, xi_grads, xi_updates, EIG, x_mean, x_std, d_sim, flow_norms, post_norms, prior_post_diff, prior_post_cont


def update_pce(
    flow_params: hk.Params,
    xi_params: hk.Params, # Note: these are passed in scaled
    prng_key: PRNGKey,
    optimizer: optax.GradientTransformation,
    opt_state: OptState,
    ema: Optional[optax.GradientTransformation],
    ema_opt_state: Optional[OptState],
    final_ys: Array,
    sde_dict: dict,
    likelihood_lp: Callable,
    N: int,
    M: int,
    theta_0: Array,
    lam: float,
    opt_round: int,
    design_min: float = 0.01,
    design_max: float = 100.,
    xi_stddev: float = 0.3,
    end_sigma: float = 0.01,
    importance_sampling: bool = False,
    use_design_dist: bool = True,
    prev_data: Optional[Array] = None,
    prev_designs: Optional[Array] = None,
    epig_log_prob_fun: Optional[Callable] = None,
    epig_sample_fun: Optional[Callable] = None,
    epig_enabled: bool = False,
    epig_policy: str = "top_k",
    epig_K: int = 8,
    epig_S: int = 1,
    epig_temperature: float = 1.0,
    epig_dropout_crn: bool = False,
) -> Tuple[hk.Params, OptState]:
    """Single SGD update step for design optimization."""
    grads, loss, conditional_lp, EIG, x_mean, x_std, d_sim, EIGs, epig_diagnostics = compute_grads_and_loss_sir_pce(
        flow_params,
        xi_params,
        prng_key,
        final_ys,
        jnp.array(sde_dict['ts'].numpy()),
        likelihood_lp,
        N,
        M,
        theta_0,
        lam,
        prev_data,
        prev_designs,
        design_min,
        design_max,
        use_design_dist,
        epig_log_prob_fun,
        epig_sample_fun,
        epig_enabled,
        epig_policy,
        epig_K,
        epig_S,
        epig_temperature,
        epig_dropout_crn,
        )
    # TODO: Make sure you just update the mu parameter
    combo_params = (flow_params, xi_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, combo_params)
    new_params, new_xi_params = optax.apply_updates(combo_params, updates)
    if ema is not None and ema_opt_state is not None:
        ema_params, new_ema_opt_state = ema.update(new_params, ema_opt_state)
    else:
        ema_params, new_ema_opt_state = new_params, ema_opt_state

    # Calculate grads of flow and post params to plot for debugging
    flow_norms = compute_average_norm(grads[0])

    # Exponential schedule for standard deviation
    decay_rate = 10.
    decay_constant = 10_000 / decay_rate
    start_sigma = xi_stddev
    # new_xi_params['xi_stddev'] = normalize_xi_to_gaussian((end_sigma + (start_sigma - end_sigma) * jnp.exp(-opt_round / decay_constant)) * 100)
    new_xi_params['xi_stddev'] = jnp.log((end_sigma + (start_sigma - end_sigma) * jnp.exp(-opt_round / decay_constant)) * 100)

    if importance_sampling:
        # Importance sampling step for designs
        # d_sim: shape [N,1]
        # EIGs: shape [N,]
        xi_mu = jnp.exp(xi_params['xi_mu'])
        xi_stddev = jnp.exp(xi_params['xi_stddev'])
        pdf_values = jax.scipy.stats.norm.pdf(
            d_sim[:,-1:], loc=xi_mu, scale=xi_stddev)
        truncation_correction = jax.scipy.stats.norm.cdf(
            100, loc=xi_mu, scale=xi_stddev) - \
            jax.scipy.stats.norm.cdf(0., loc=xi_mu, scale=xi_stddev)
        adjusted_pdf_values = pdf_values / truncation_correction
        design_lps = jnp.log(adjusted_pdf_values)
        EIGs_shifted = jnp.log(EIGs - jnp.min(EIGs) + 1e-6)
        EIGs_max_shifted = EIGs_shifted - jnp.max(EIGs_shifted)
        w_i = design_lps.squeeze() + EIGs_max_shifted
        p_tilde = jax.nn.softmax(w_i)
        E_p_tilde_xi = jnp.sum(d_sim[:,-1] * p_tilde)
        new_xi_params['xi_mu'] = jnp.log(
            (jnp.exp(new_xi_params['xi_mu']) + E_p_tilde_xi) / 2)
        # # debug the EIGs
        # plot_eig_histogram(d_sim.squeeze(), EIGs, './')
        # breakpoint()
    xi_grads = grads[1]
    xi_updates = updates[1]

    return new_params, new_xi_params, new_opt_state, loss, grads, xi_grads, xi_updates, EIG, x_mean, x_std, d_sim, flow_norms, conditional_lp, ema_params, new_ema_opt_state, epig_diagnostics
