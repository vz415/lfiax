import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from jax.scipy.special import logsumexp

from functools import partial
import distrax
import haiku as hk

from lfiax.utils.simulators import sim_linear_prior, sim_linear_data_vmap, sim_linear_prior_M_samples, simulate_sir
from lfiax.utils.utils import get_calibration_error_jax


from typing import Any, Callable, NamedTuple, Tuple

Array = jnp.ndarray
PRNGKey = Array


@jax.jit
def pairwise_distances(points):
    """
    Calculates the pairwise distances between a set of points.

    Args:
        points: an array of shape (n_points, n_dims) containing the coordinates of the points

    Returns:
        dists: an array of shape (n_points, n_points) containing the pairwise distances between the points
    """
    n_dims, n_points = points.shape
    tiled_points = jnp.tile(points, (1, n_points, 1))
    transposed_points = jnp.transpose(tiled_points, axes=(0, 2, 1))
    diffs = tiled_points - transposed_points
    return diffs

@jax.jit
def measure_of_spread(points):
    """
    Calculates a measure of spread for a set of points.

    Args:
        points: an array of shape (n_points, n_dims) containing the coordinates of the points

    Returns:
        spread: a scalar value indicating the spread of the points
    """
    dists = jnp.abs(jnp.subtract(points, points.T))
    cov = jnp.cov(dists)  # , rowvar=False)
    eigvals = jnp.linalg.eigvalsh(cov)
    spread = jnp.sum(jnp.sqrt(jnp.maximum(eigvals, 0.0)))
    return spread

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

    scaled_x = jax.lax.cond(x.shape[-1] == 1, single_column_fn, multi_column_fn, x)
    return scaled_x

def _safe_mean_terms(terms):
    mask = jnp.isnan(terms) | (terms == -jnp.inf) | (terms == jnp.inf)
    nonnan = jnp.sum(~mask, axis=0, dtype=jnp.float32)
    terms = jnp.where(mask, 0.0, terms)
    loss = terms / nonnan
    agg_loss = jnp.sum(loss)
    return agg_loss, loss

@partial(jax.jit, static_argnums=[3, 5, 6])
def lf_pce_eig_scan_lin_reg(
    flow_params: hk.Params,
    xi_params: hk.Params,
    prng_key: PRNGKey,
    log_prob_fun: Callable,
    designs: Array,
    N: int = 100,
    M: int = 10,
    lam: float = 0.5,
):
    """
    Calculates LF-PCE loss using jax.lax.scan to accelerate.
    """

    def compute_marginal_lp(keys, log_prob_fun, M, N, x, conditional_lp):
        def scan_fun(contrastive_lps, i):
            theta, _ = sim_linear_prior(N, keys[i + 1])
            contrastive_lp = log_prob_fun(flow_params, x, theta, xi)
            return jnp.logaddexp(contrastive_lps, contrastive_lp), i + 1

        result = jax.lax.scan(scan_fun, conditional_lp, jnp.array(range(M)))
        return result[0]

    keys = jrandom.split(prng_key, 1 + M)

    xi = jnp.broadcast_to(xi_params["xi"], (N, xi_params["xi"].shape[-1]))

    # simulate the outcomes before finding their log_probs
    # `designs` are combos of previous designs and proposed (non-scaled) designs
    x, theta_0, x_noiseless, noise = sim_linear_data_vmap(designs, N, keys[0])

    scaled_x = standard_scale(x)
    x_mean, x_std = jnp.mean(x), jnp.std(x) + 1e-10
    # If this is the wrong shape, grads don't flow :(
    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)

    conditional_lp = log_prob_fun(flow_params, scaled_x, theta_0, xi)
    marginal_lp = compute_marginal_lp(
        keys[1 : M + 1], log_prob_fun, M, N, scaled_x, conditional_lp
    ) - jnp.log(M + 1)

    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    loss = EIG + lam * jnp.mean(conditional_lp)

    return -loss, (conditional_lp, theta_0, x, x_noiseless, noise, EIG, x_mean, x_std)

@partial(jax.jit, static_argnums=[3, 6, 7, 8])
def lf_pce_eig_scan(
    flow_params: hk.Params,
    xi_params: hk.Params,
    prng_key: PRNGKey,
    prior: Callable,
    scaled_x: Array,
    theta_0: Array,
    log_prob_fun: Callable,
    N: int = 100,
    M: int = 10,
    lam: float = 0.5,
):
    """
    Calculates LF-PCE loss using jax.lax.scan to accelerate.
    """

    def compute_marginal_lp(keys, log_prob_fun, M, N, x, conditional_lp):
        def scan_fun(contrastive_lps, i):
            # TODO: Make sample_shape adapt to passed prior instead of pre-specified shape.
            theta = prior.sample(seed=keys[i + 1], sample_shape=(N, 2))
            contrastive_lp = log_prob_fun(flow_params, x, theta, xi)
            return jnp.logaddexp(contrastive_lps, contrastive_lp), i + 1

        result = jax.lax.scan(scan_fun, conditional_lp, jnp.array(range(M)))
        return result[0]

    keys = jrandom.split(prng_key, 2 + M)

    # Broadcast xi design params & initial priors
    xi = jnp.broadcast_to(xi_params["xi"], (N, xi_params["xi"].shape[-1]))

    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)

    conditional_lp = log_prob_fun(flow_params, scaled_x, theta_0, xi)
    marginal_lp = compute_marginal_lp(
        keys[1 : M + 1], log_prob_fun, M, N, scaled_x, conditional_lp
    ) - jnp.log(M + 1)

    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    loss = EIG + lam * jnp.mean(conditional_lp)

    return -loss, (conditional_lp, EIG)

@partial(jax.jit, static_argnums=[3, 6, 7, 8])
def snpe_c(
    post_params: hk.Params,
    xi_params: hk.Params,
    prng_key: PRNGKey,
    prior: Callable,
    scaled_x: Array,
    theta_0: Array,
    post_log_prob_fun: Callable,
    N: int = 100,
    M: int = 10,
    lam: float = 0.5,
):
    """
    Calculates NP-PCE loss using jax.lax.scan to accelerate. Requires a likelihood
    log_prob function and a prior. Will use to calculate the EIG and amortized density.
    """

    def compute_snpe_marginal_lp(
        keys, prior, post_log_prob_fun, M, N, x, conditional_lp
    ):
        def scan_fun(contrastive_lps, i):
            # TODO: Make sample_shape adapt to passed prior instead of pre-specified shape.
            # TODO: Conditional statement if prior is a flow
            thetas, prior_lp = prior.sample_and_log_prob(
                seed=keys[i + 1], sample_shape=(N,)
            )
            contrastive_lp = post_log_prob_fun(post_params, thetas, x, xi)
            contrastive_lp = contrastive_lp - prior_lp
            return jnp.logaddexp(contrastive_lps, contrastive_lp), i + 1

        result = jax.lax.scan(scan_fun, conditional_lp, jnp.array(range(M)))
        return result[0]

    keys = jrandom.split(prng_key, 2 + M)

    # Broadcast scaled xi design params & initial priors
    xi = jnp.broadcast_to(xi_params["xi"], (N, xi_params["xi"].shape[-1]))

    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)

    conditional_lp = post_log_prob_fun(post_params, theta_0, scaled_x, xi)
    prior_lp = prior.log_prob(theta_0)
    conditional_lp = conditional_lp - prior_lp
    marginal_lp = compute_snpe_marginal_lp(
        keys[1 : M + 1], prior, post_log_prob_fun, M, N, scaled_x, conditional_lp
    ) - jnp.log(M + 1)

    # EIG = jnp.sum(conditional_lp - marginal_lp)
    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    loss = EIG + lam * jnp.mean(conditional_lp - prior_lp)

    return -loss, (conditional_lp, EIG)

@partial(jax.jit, static_argnums=[4, 5, 6, 7, 9, 10])
def lf_ace_eig_scan(
    flow_params: hk.Params,
    post_params: hk.Params,
    xi_params: hk.Params,
    prng_key: PRNGKey,
    scaled_x: Array,
    theta_0: Array,
    prior: Callable,
    log_prob_fun: Callable,
    post_log_prob_fun: Callable,
    post_sample_fun: Callable,
    designs: Array,
    N: int = 100,
    M: int = 10,
):
    """
    *Work in progress.*
    Calculates snpe-c using a posterior and prior. Requires a posterior and prior
    estimate. Will use all three to calculate the EIG. This takes a vectorized
    approach for readability and GPU compatability.
    """

    def compute_snpe_marginal_lp(keys, log_prob_fun, M, N, x, conditional_lp):
        def scan_fun(contrastive_lps, i):
            # TODO: Make sample_shape adapt to passed prior instead of pre-specified shape.
            theta = post_sample_fun.sample(seed=keys[i + 1], sample_shape=(N, 2))
            contrastive_lp = log_prob_fun(flow_params, x, theta, xi)
            # TODO conditional statement if prior is a flow
            prior_lp = prior.log_prob(theta)
            numerator = jnp.logaddexp(prior_lp, contrastive_lp)
            post_lp = post_log_prob_fun(post_params, theta, x)
            contrastive_lp = jnp.logaddexp(numerator, -post_lp)
            return jnp.logaddexp(contrastive_lps, contrastive_lp), i + 1

        result = jax.lax.scan(scan_fun, conditional_lp, jnp.array(range(M)))
        return result[0]

    keys = jrandom.split(prng_key, 2 + 3 * M)

    # Broadcast xi design params & initial priors
    xi = jnp.broadcast_to(xi_params["xi"], (N, xi_params["xi"].shape[-1]))

    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)

    conditional_lp = log_prob_fun(flow_params, scaled_x, theta_0, xi)
    marginal_lp = compute_snpe_marginal_lp(
        keys[1 : M + 1], log_prob_fun, M, N, scaled_x, conditional_lp
    ) - jnp.log(M + 1)

    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    loss = EIG + jnp.mean(conditional_lp)

    return -loss, (conditional_lp, EIG)

@partial(jax.jit, static_argnums=[2, 3])
def lfi_pce_eig_vmap_distrax(
    params: hk.Params, prng_key: PRNGKey, N: int = 100, M: int = 10, **kwargs
):
    """
    *Work in progress.*
    Calculates PCE loss using vmap inherent to `distrax` distributions. May be faster
    than scan on GPUs.
    TODO: refactor arguments.
    """
    keys = jrandom.split(prng_key, 2)
    xi = params["xi"]
    flow_params = {k: v for k, v in params.items() if k != "xi"}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])

    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    # TODO: Make function that returns M x num_samples priors
    thetas, log_probs = sim_linear_prior_M_samples(
        num_samples=num_samples, M=M, key=keys[1]
    )

    # conditional_lp could be the initial starting state that is added upon...
    contrastive_lps = jax.vmap(
        lambda theta: log_prob.apply(params, x, theta, xi_broadcast)
    )(thetas)
    marginal_log_prbs = jnp.concatenate(
        (jax_lexpand(conditional_lp, 1), jnp.array(contrastive_lps))
    )
    marginal_lp = jax.nn.logsumexp(marginal_log_prbs, 0) - math.log(M + 1)
    # marginal_lp = compute_marginal_lp3(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp)

    return -sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)

@partial(jax.jit, static_argnums=[2, 3])
def lfi_pce_eig_vmap_manual(
    params: hk.Params, prng_key: PRNGKey, N: int = 100, M: int = 10, **kwargs
):
    """
    *Work in progress.*
    Calculates PCE loss using explicit vmap of `distrax` distributions. May potentially
    be more stable than using `ditrax` implicit version as of 2/9/23. May be faster
    than scan on GPUs.
    TODO: refactor arguments.
    """
    keys = jrandom.split(prng_key, M + 1)
    xi = params["xi"]
    flow_params = {k: v for k, v in params.items() if k != "xi"}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])

    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    thetas, log_probs = jax.vmap(partial(sim_linear_prior, num_samples))(
        keys[1 : M + 1]
    )

    # conditional_lp could be the initial starting state that is added upon...
    contrastive_lps = jax.vmap(
        lambda theta: log_prob.apply(params, x, theta, xi_broadcast)
    )(thetas)
    marginal_log_prbs = jnp.concatenate(
        (jax_lexpand(conditional_lp, 1), jnp.array(contrastive_lps))
    )
    marginal_lp = jax.nn.logsumexp(marginal_log_prbs, 0) - math.log(M + 1)
    # marginal_lp = compute_marginal_lp3(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp)

    return -sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)

# SIR and EPIG experiment helpers.

class EpigDiagnostics(NamedTuple):
    score_mean: Array
    score_max: Array
    selected_score_mean: Array
    effective_sample_size: Array
    max_importance_weight: Array
    valid_score_fraction: Array
    selected_valid_fraction: Array
    fallback_used: Array
    sample_finite_fraction: Array
    log_prob_finite_fraction: Array
    ratio_finite_fraction: Array
    sample_abs_max: Array
    log_prob_abs_max: Array
    centered_log_prob_abs_max: Array
    ratio_abs_max: Array

class EpigScanDiagnostics(NamedTuple):
    sample_finite_fraction: Array
    log_prob_finite_fraction: Array
    ratio_finite_fraction: Array
    sample_abs_max: Array
    log_prob_abs_max: Array
    centered_log_prob_abs_max: Array
    ratio_abs_max: Array

def _finite_mean(values: Array, finite: Array) -> Array:
    count = jnp.sum(finite)
    total = jnp.sum(jnp.where(finite, values, 0.0))
    return jnp.where(count > 0, total / count, jnp.nan)

def _finite_max(values: Array, finite: Array) -> Array:
    maximum = jnp.max(jnp.where(finite, values, -jnp.inf))
    return jnp.where(jnp.any(finite), maximum, jnp.nan)

def _center_component_log_probs(log_probs: Array) -> Array:
    """Remove per-sample component offsets that cancel from the EPIG ratio."""
    offsets = jnp.max(log_probs, axis=0, keepdims=True)
    finite_offsets = jnp.where(jnp.isfinite(offsets), offsets, 0.0)
    return log_probs - jax.lax.stop_gradient(finite_offsets)


@jax.jit
def shuffle_samples(key, x, theta, xi):
    num_samples = x.shape[0]
    shuffled_indices = jax.random.permutation(key, num_samples)
    return x[shuffled_indices], theta[shuffled_indices], xi[shuffled_indices]


@partial(
    jax.jit,
    static_argnums=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22],
)
def lf_pce_design_dist_sir(
    flow_params: hk.Params,
    xi_params_scaled: hk.Params,
    prng_key: PRNGKey,
    final_ys: Array,
    sde_dict_ts: Array,
    theta_0: Array,
    prev_data: Array = None,
    prev_designs: Array = None,
    log_prob_fun: Callable = None,
    N: int=100,
    M: int=10,
    lam: float=0.5,
    design_min: float=0.01,
    design_max: float=100.,
    use_design_dist: bool=True,
    epig_log_prob_fun: Callable = None,
    epig_sample_fun: Callable = None,
    epig_enabled: bool = False,
    epig_policy: str = "top_k",
    epig_K: int = 8,
    epig_S: int = 1,
    epig_temperature: float = 1.0,
    epig_dropout_crn: bool = False,
    ):
    """
    Calculates LF-PCE loss using jax.lax.scan to accelerate. Takes the previously
    simulated outputs from designs, broadcasts the xi design params, and uses
    the log_prob_fun to calculate the EIG and amortized density.

    When EPIG is enabled, the incoming ``theta_0``/SDE pairs form a candidate
    pool. EPIG selects the N paired positives used by LF-PCE.
    """
    design_key, epig_key, policy_key, shuffle_key, likelihood_key = jrandom.split(
        prng_key, 5
    )
    likelihood_keys = jrandom.split(likelihood_key, 1 + M)
    candidate_count = theta_0.shape[0]

    # Unnormalize designs to get proper output values
    if use_design_dist:
        xi_params = {k: jnp.exp(v) for k, v in xi_params_scaled.items() if k in ['xi_mu', 'xi_stddev']}
        # xi_params = {k: inverse_normalize_xi(v) for k, v in xi_params_scaled.items() if k in ['xi_mu', 'xi_stddev']}
        a, b = (design_min - xi_params['xi_mu']) / xi_params['xi_stddev'], (design_max - xi_params['xi_mu']) / xi_params['xi_stddev']
        d_sim = xi_params['xi_mu'] + xi_params['xi_stddev'] * jrandom.truncated_normal(
            design_key, a, b, shape=(candidate_count, 1)
        )
    else:
        d_sim = jnp.broadcast_to(xi_params_scaled['xi_mu'], (candidate_count, 1))

    def total_log_prob(params, key, x, theta, designs):
        if x.shape[1] == 1:
            return log_prob_fun(params, key, x, theta, designs)
        dropout_keys = jrandom.split(key, num=x.shape[1])
        per_design_lp = jax.vmap(
            log_prob_fun, in_axes=(None, 0, -1, None, -1)
        )(params, dropout_keys, x[:, jnp.newaxis], theta, designs[:, jnp.newaxis])
        return jnp.sum(per_design_lp, axis=0)

    nan = jnp.array(jnp.nan, dtype=d_sim.dtype)
    diagnostics = EpigDiagnostics(*([nan] * len(EpigDiagnostics._fields)))

    if epig_enabled:
        _, raw_epig_scores, epig_scan_diagnostics = lf_epig_scan(
            flow_params,
            d_sim,
            epig_key,
            theta_0,
            theta_0,
            epig_log_prob_fun,
            epig_sample_fun,
            K=epig_K,
            S=epig_S,
            paired_designs=True,
            return_diagnostics=True,
            dropout_crn=epig_dropout_crn,
        )
        epig_score_finite = jnp.isfinite(raw_epig_scores)
        valid_score_count = jnp.sum(epig_score_finite)
        epig_scores = jnp.where(epig_score_finite, raw_epig_scores, -1e30)
        fallback_used = valid_score_count < N

        if epig_policy == "top_k":
            _, selected_indices = jax.lax.top_k(epig_scores, N)
            effective_sample_size = jnp.asarray(N, dtype=d_sim.dtype)
            max_importance_weight = jnp.array(1.0, dtype=d_sim.dtype)
        elif epig_policy == "softmax_corrected":
            selection_probs = jax.nn.softmax(epig_scores / epig_temperature)
            selected_indices = jrandom.choice(
                policy_key,
                candidate_count,
                shape=(N,),
                replace=True,
                p=selection_probs,
            )
            selected_log_weights = (
                -jnp.log(candidate_count)
                - jnp.log(selection_probs[selected_indices] + 1e-30)
            )
            selected_weights = jnp.exp(selected_log_weights)
            effective_sample_size = (
                jnp.sum(selected_weights) ** 2
                / (jnp.sum(selected_weights ** 2) + 1e-30)
            )
            max_importance_weight = jnp.max(selected_weights)
        else:
            raise ValueError(f"Unknown EPIG selection policy: {epig_policy}")

        selected_indices = jax.lax.select(
            fallback_used, jnp.arange(N), selected_indices
        )
        selected_scores = raw_epig_scores[selected_indices]
        selected_score_finite = epig_score_finite[selected_indices]
        diagnostics = EpigDiagnostics(
            _finite_mean(raw_epig_scores, epig_score_finite),
            _finite_max(raw_epig_scores, epig_score_finite),
            _finite_mean(selected_scores, selected_score_finite),
            jnp.where(
                fallback_used,
                jnp.asarray(N, d_sim.dtype),
                effective_sample_size,
            ),
            jnp.where(
                fallback_used,
                jnp.array(1.0, d_sim.dtype),
                max_importance_weight,
            ),
            jnp.mean(epig_score_finite),
            jnp.mean(selected_score_finite),
            fallback_used.astype(d_sim.dtype),
            jnp.mean(epig_scan_diagnostics.sample_finite_fraction),
            jnp.mean(epig_scan_diagnostics.log_prob_finite_fraction),
            jnp.mean(epig_scan_diagnostics.ratio_finite_fraction),
            jnp.max(epig_scan_diagnostics.sample_abs_max),
            jnp.max(epig_scan_diagnostics.log_prob_abs_max),
            jnp.max(epig_scan_diagnostics.centered_log_prob_abs_max),
            jnp.max(epig_scan_diagnostics.ratio_abs_max),
        )
        theta_0 = theta_0[selected_indices]
        final_ys = final_ys[:, selected_indices]
        d_sim = d_sim[selected_indices]
        if prev_data is not None:
            prev_data = prev_data[selected_indices]
            prev_designs = prev_designs[selected_indices]

    sim_x, x_mean, x_std = simulate_sir(d_sim, sde_dict_ts, final_ys)
    xi = d_sim

    if len(sim_x.shape) > 2:
        sim_x = sim_x.squeeze(0)

    if prev_data is not None:
        x_combined = jnp.concatenate([prev_data, sim_x], axis=1)
        xi = jnp.concatenate([prev_designs, xi], axis=1)
    else:
        x_combined = sim_x

    x_combined, theta_0, xi = shuffle_samples(
        shuffle_key, x_combined, theta_0, xi
    )

    conditional_lp = total_log_prob(
        flow_params, likelihood_keys[0], x_combined, theta_0, xi
    )

    def compute_marginal_lp(params, theta_seed):
        def scan_fun(carry, i):
            contrastive_lps, theta_i = carry
            theta_i = jnp.roll(theta_i, shift=1, axis=0)
            contrastive_lp = total_log_prob(
                params, likelihood_keys[i + 1], x_combined, theta_i, xi
            )
            return (jnp.logaddexp(contrastive_lps, contrastive_lp), theta_i), None

        result = jax.lax.scan(
            scan_fun, (conditional_lp, theta_seed), jnp.arange(M)
        )
        return result[0][0] - jnp.log(M + 1)

    marginal_lp = compute_marginal_lp(flow_params, theta_0)

    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    loss = EIG + lam * jnp.mean(conditional_lp)

    return -loss, (conditional_lp, EIG, EIGs, x_mean, x_std, xi, diagnostics)

@partial(jax.jit, static_argnums=[7,8,9,10,11,12,13,14,15,16])
def lf_ace_design_dist_sir(flow_params: hk.Params,
                           post_params: hk.Params,
                           xi_params_scaled: hk.Params,
                           prng_key: PRNGKey,
                           final_ys: Array,
                            sde_dict_ts: Array,
                            theta_0: Array,
                            likelihood_lp_fun: Callable,
                            prior_lp_fun: Callable,
                            prior_sample_fun: Callable,
                            post_lp_fun: Callable,
                            post_sample_fun: Callable,
                            N: int=100,
                            M: int=10,
                            lam: float=0.5,
                            design_min: float=0.01,
                            design_max: float=100.,
                            sbc_samples: int=32,
                            sbc_lambda: float=1.,
                            ):
    """
    Calculates LF-ACE loss using jax.lax.scan to accelerate. Only requires a likelihood
    and posterior.

    The "prior_lp_fun" needs to be passed in with the
    """
    keys = jrandom.split(prng_key, 3 + M)

    # Unnormalize designs to get proper output values
    xi_params = {k: jnp.multiply(v, 100.) for k, v in xi_params_scaled.items() if k in ['xi_mu', 'xi_stddev']}
    a, b = (design_min - xi_params['xi_mu']) / xi_params['xi_stddev'], (design_max - xi_params['xi_mu']) / xi_params['xi_stddev']
    d_sim = xi_params['xi_mu'] + xi_params['xi_stddev'] * jrandom.truncated_normal(keys[0], a, b, shape=(N,1))

    # Scale for conditional flow
    xi = d_sim/100.

    scaled_x, x_mean, x_std = simulate_sir(d_sim,
                                           sde_dict_ts,
                                           final_ys/100.)

    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)

    def compute_marginal_lp(keys, M, theta_0, x, conditional_lp):
        def scan_fun(carry, i):
            contrastive_lps = carry
            theta, _ = post_sample_fun(post_params, keys[i], N, x)
            contrastive_prior = prior_lp_fun(theta)
            contrastive_likelihood = likelihood_lp_fun(flow_params, x, theta, xi)
            contrastive_posterior = post_lp_fun(
                # jax.lax.stop_gradient(post_params), theta, x)
                post_params, theta, x)
            contrastive_lp = (contrastive_prior + contrastive_likelihood) - contrastive_posterior
            contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
            return (contrastive_lp), i + 1

        contrastive_prior = prior_lp_fun(theta_0)
        contrastive_likelihood = likelihood_lp_fun(flow_params, x, theta_0, xi)
        contrastive_posterior = post_lp_fun(
            # jax.lax.stop_gradient(post_params), theta_0, x)
            post_params, theta_0, x)
        conditional_lp = contrastive_prior + contrastive_likelihood - contrastive_posterior
        initial_carry = (conditional_lp)

        result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
        return result[0]

    conditional_lp = likelihood_lp_fun(flow_params, scaled_x, theta_0, xi)

    # BUG: This should be shape [512,1]
    marginal_lp = compute_marginal_lp(
        keys[1:M+1], M, theta_0, scaled_x, conditional_lp
        ) - jnp.log(M + 1)

    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    loss = EIG + lam * jnp.mean(conditional_lp)# + sbc_lambda * sbc_regularization

    theta, contrastive_posterior = post_sample_fun(post_params, keys[0], N, scaled_x)
    prior_lp = prior_lp_fun(theta_0)
    contrastive_prior = prior_lp_fun(theta)
    contrastive_likelihood = likelihood_lp_fun(flow_params, scaled_x, theta_0, xi)
    posterior_lp = post_lp_fun(
        jax.lax.stop_gradient(post_params), theta_0, scaled_x)
    contrastive_lp = (prior_lp + contrastive_likelihood) - posterior_lp
    prior_post_diff = prior_lp - posterior_lp
    prior_post_cont = contrastive_prior - contrastive_posterior
    # best_design_i = jnp.argmax(EIGs)
    # jax.debug.breakpoint()

    return -loss , (conditional_lp, EIG, EIGs, x_mean, x_std, d_sim, prior_post_diff, prior_post_cont)


def _logmeanexp(a: Array, axis: int) -> Array:
    return logsumexp(a, axis=axis) - jnp.log(a.shape[axis])

@partial(jax.jit, static_argnums=[5, 6, 7, 8, 9, 10, 11])
def lf_epig_scan(
    flow_params: hk.Params,
    static_designs: Array,
    prng_key: PRNGKey,
    theta_proposals: Array,    # (Q, theta_dim) or (theta_dim,)
    theta_targets: Array,      # (J, theta_dim) or (theta_dim,)
    log_prob_fun: Callable,    # (params, y, theta, xi, dropout_key) -> log p(y|theta,xi)
    sample_fun: Callable,      # (..., sample_keys, dropout_key) -> (R, y_dim...)
    K: int = 32,               # number of dropout "model particles" phi^(k)
    S: int = 1,                # requested base/sample keys per model particle
    paired_designs: bool = False,
    return_diagnostics: bool = False,
    dropout_crn: bool = False,
) -> Any:
    """
    EPIG acquisition score for simulator-parameter active learning using ONLY likelihood evals,
    with epistemic uncertainty approximated via MC-dropout.

    EPIG(theta) = E_{theta* ~ p*(theta*)} KL( p(y,y*|theta,theta*) || p(y|theta)p(y*|theta*) )

    Dropout key indexes the model particle: phi^(k).
    We sample (y, y*) from the JOINT predictive by using the *same* dropout particle k for both.

    Q and J are scanned to bound peak memory, while K and each sampler-returned
    batch remain vectorized. The sampler owns how the requested S base draws
    are constructed and may return a different static batch size R (for
    example, 2 * S antithetic samples). When ``paired_designs`` is true,
    ``static_designs[i]`` is used only for ``theta_proposals[i]``. Otherwise
    every proposal shares ``static_designs``. When ``dropout_crn`` is true,
    every proposal reuses the same K dropout-particle keys; query and target
    base-noise keys remain proposal-specific.

    Returns:
        loss:    -EPIG scores (so you can minimize)
        scores:  EPIG scores
        diagnostics: optional per-proposal finite fractions for samples,
            component log-probabilities, and density ratios
    """
    # Normalize shapes
    squeeze_proposal = False
    if theta_proposals.ndim == 1:
        theta_proposals = theta_proposals[jnp.newaxis, :]
        squeeze_proposal = True
    if theta_targets.ndim == 1:
        theta_targets = theta_targets[jnp.newaxis, :]

    Q = theta_proposals.shape[0]
    J = theta_targets.shape[0]
    if Q < 1 or J < 1:
        raise ValueError("EPIG requires at least one proposal and one target.")

    if dropout_crn:
        key_drop, key_proposals = jrandom.split(prng_key, 2)
        shared_drop_keys = jrandom.split(key_drop, K)
        proposal_keys = jrandom.split(key_proposals, Q)
    else:
        # Preserve the original key schedule for backwards-compatible runs.
        shared_drop_keys = None
        proposal_keys = jrandom.split(prng_key, Q)

    def epig_one_proposal(theta_q: Array, xi: Array, key_q: PRNGKey) -> Array:
        if dropout_crn:
            # Dropout particles are common random numbers across proposals.
            # Only flow/base sampling noise remains proposal-specific.
            key_y, key_t = jrandom.split(key_q, 2)
            drop_keys = shared_drop_keys
        else:
            # Original behavior: each proposal gets an independent ensemble.
            key_drop, key_y, key_t = jrandom.split(key_q, 3)
            drop_keys = jrandom.split(key_drop, K)

        # --- Sample y from each dropout particle at theta_q ---
        y_keys = jrandom.split(key_y, K * S).reshape(K, S, 2)

        def _sample_batch(
            theta: Array, drop_key: PRNGKey, sample_keys: Array
        ) -> Array:
            return sample_fun(flow_params, theta, xi, sample_keys, drop_key)

        # y_samples: (K,R,y_dim...), where R is owned by sample_fun.
        y_samples = jax.vmap(
            lambda dk, sk: _sample_batch(theta_q, dk, sk)
        )(drop_keys, y_keys)
        G = y_samples.shape[0] * y_samples.shape[1]
        y_flat = y_samples.reshape((G,) + y_samples.shape[2:])
        y_sample_finite = jnp.mean(jnp.isfinite(y_flat))

        # logp_y: (K_components, G_samples)
        def _lp_component_y(drop_key_component: PRNGKey) -> Array:
            return jax.vmap(lambda y: log_prob_fun(flow_params, y, theta_q, xi, drop_key_component))(y_flat)

        logp_y = jax.vmap(_lp_component_y)(drop_keys)  # (K, G)
        centered_logp_y = _center_component_log_probs(logp_y)
        log_marg_y = _logmeanexp(centered_logp_y, axis=0)  # (G,)

        # --- For each target theta*, estimate KL(joint || product) using the same dropout particles ---
        target_keys = jrandom.split(key_t, J)  # one RNG per target (controls only base-noise for y*)

        def kl_one_target(theta_star: Array, key_star: PRNGKey) -> Array:
            # Sample y* from each dropout particle at theta_star
            ystar_keys = jrandom.split(key_star, K * S).reshape(K, S, 2)
            ystar_samples = jax.vmap(
                lambda dk, sk: _sample_batch(theta_star, dk, sk)
            )(drop_keys, ystar_keys)
            ystar_count = ystar_samples.shape[0] * ystar_samples.shape[1]
            if ystar_count != G:
                raise ValueError(
                    "EPIG sampler must return the same batch size for query "
                    "and target conditioning values."
                )
            ystar_flat = ystar_samples.reshape(
                (ystar_count,) + ystar_samples.shape[2:]
            )
            ystar_sample_finite = jnp.mean(jnp.isfinite(ystar_flat))

            # logp_star: (K_components, G_samples)
            def _lp_component_star(drop_key_component: PRNGKey) -> Array:
                return jax.vmap(lambda y: log_prob_fun(flow_params, y, theta_star, xi, drop_key_component))(ystar_flat)

            logp_star = jax.vmap(_lp_component_star)(drop_keys)  # (K, G)
            centered_logp_star = _center_component_log_probs(logp_star)

            log_marg_star = _logmeanexp(centered_logp_star, axis=0)  # (G,)
            log_joint = _logmeanexp(
                centered_logp_y + centered_logp_star, axis=0
            )

            # MC estimate of KL via samples from joint predictive
            r = log_joint - log_marg_y - log_marg_star           # (G,)
            ratio_finite = jnp.isfinite(r)
            sample_finite_fraction = 0.5 * (
                y_sample_finite + ystar_sample_finite
            )
            log_prob_finite_fraction = 0.5 * (
                jnp.mean(jnp.isfinite(logp_y))
                + jnp.mean(jnp.isfinite(logp_star))
            )
            ratio_finite_fraction = jnp.mean(ratio_finite)
            sample_abs_max = jnp.maximum(
                jnp.max(jnp.where(jnp.isfinite(y_flat), jnp.abs(y_flat), 0.0)),
                jnp.max(
                    jnp.where(jnp.isfinite(ystar_flat), jnp.abs(ystar_flat), 0.0)
                ),
            )
            log_prob_abs_max = jnp.maximum(
                jnp.max(jnp.where(jnp.isfinite(logp_y), jnp.abs(logp_y), 0.0)),
                jnp.max(
                    jnp.where(jnp.isfinite(logp_star), jnp.abs(logp_star), 0.0)
                ),
            )
            centered_log_prob_abs_max = jnp.maximum(
                jnp.max(
                    jnp.where(
                        jnp.isfinite(centered_logp_y),
                        jnp.abs(centered_logp_y),
                        0.0,
                    )
                ),
                jnp.max(
                    jnp.where(
                        jnp.isfinite(centered_logp_star),
                        jnp.abs(centered_logp_star),
                        0.0,
                    )
                ),
            )
            ratio_abs_max = jnp.max(jnp.where(jnp.isfinite(r), jnp.abs(r), 0.0))
            return (
                _finite_mean(r, ratio_finite),
                sample_finite_fraction,
                log_prob_finite_fraction,
                ratio_finite_fraction,
                sample_abs_max,
                log_prob_abs_max,
                centered_log_prob_abs_max,
                ratio_abs_max,
            )

        zero = jnp.zeros((), dtype=log_marg_y.dtype)
        initial_target_carry = (zero,) * 9

        def scan_target(carry, target_args):
            (
                score_sum,
                score_count,
                sample_finite_sum,
                log_prob_finite_sum,
                ratio_finite_sum,
                sample_abs_max,
                log_prob_abs_max,
                centered_log_prob_abs_max,
                ratio_abs_max,
            ) = carry
            (
                target_score,
                sample_finite,
                log_prob_finite,
                ratio_finite,
                target_sample_abs_max,
                target_log_prob_abs_max,
                target_centered_log_prob_abs_max,
                target_ratio_abs_max,
            ) = kl_one_target(*target_args)
            target_is_finite = jnp.isfinite(target_score)
            return (
                score_sum + jnp.where(target_is_finite, target_score, 0.0),
                score_count + target_is_finite.astype(log_marg_y.dtype),
                sample_finite_sum + sample_finite,
                log_prob_finite_sum + log_prob_finite,
                ratio_finite_sum + ratio_finite,
                jnp.maximum(sample_abs_max, target_sample_abs_max),
                jnp.maximum(log_prob_abs_max, target_log_prob_abs_max),
                jnp.maximum(
                    centered_log_prob_abs_max,
                    target_centered_log_prob_abs_max,
                ),
                jnp.maximum(ratio_abs_max, target_ratio_abs_max),
            ), None

        target_summary, _ = jax.lax.scan(
            scan_target,
            initial_target_carry,
            (theta_targets, target_keys),
        )
        (
            score_sum,
            score_count,
            sample_finite_sum,
            log_prob_finite_sum,
            ratio_finite_sum,
            sample_abs_max,
            log_prob_abs_max,
            centered_log_prob_abs_max,
            ratio_abs_max,
        ) = target_summary
        return (
            jnp.where(score_count > 0, score_sum / score_count, jnp.nan),
            EpigScanDiagnostics(
                sample_finite_sum / J,
                log_prob_finite_sum / J,
                ratio_finite_sum / J,
                sample_abs_max,
                log_prob_abs_max,
                centered_log_prob_abs_max,
                ratio_abs_max,
            ),
        )

    if paired_designs:
        if static_designs.shape[0] != Q:
            raise ValueError(
                "Paired EPIG designs must have one row per theta proposal."
            )
        _, (scores, diagnostics) = jax.lax.scan(
            lambda carry, args: (
                carry,
                epig_one_proposal(args[0], args[1], args[2]),
            ),
            None,
            (theta_proposals, static_designs, proposal_keys),
        )
    else:
        _, (scores, diagnostics) = jax.lax.scan(
            lambda carry, args: (
                carry,
                epig_one_proposal(args[0], static_designs, args[1]),
            ),
            None,
            (theta_proposals, proposal_keys),
        )

    if squeeze_proposal:
        scores = scores.squeeze(0)
        diagnostics = jax.tree_util.tree_map(lambda value: value.squeeze(0), diagnostics)

    if return_diagnostics:
        return -scores, scores, diagnostics
    return -scores, scores
