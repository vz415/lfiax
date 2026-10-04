from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from lfiax.utils.oed_losses import lf_pce_design_dist_sir

from typing import (
    Any,
    Mapping,
    Optional,
    Tuple,
    Callable
)

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


def compute_average_norm(grads):
    norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
    flat_norms = jax.tree_util.tree_leaves(norms)
    return jnp.mean(jnp.array(flat_norms))


@partial(jax.jit, static_argnames=('likelihood_lp', 'N', 'M', 'lam', 'design_min', 'design_max', 'epig_log_prob_fun', 'epig_sample_fun', 'epig_enabled', 'epig_policy', 'epig_K', 'epig_S', 'epig_dropout_crn'))
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

    xi_grads = grads[1]
    xi_updates = updates[1]

    return new_params, new_xi_params, new_opt_state, loss, grads, xi_grads, xi_updates, EIG, x_mean, x_std, d_sim, flow_norms, conditional_lp, ema_params, new_ema_opt_state, epig_diagnostics
