import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom

from functools import partial
import distrax
import haiku as hk

from lfiax.utils.simulators import sim_linear_prior, sim_linear_data_vmap, sim_linear_prior_M_samples

from typing import Any, Callable

Array = jnp.ndarray
PRNGKey = Array


def _safe_mean_terms(terms):
    # breakpoint()
    mask = jnp.isnan(terms) | (terms == -jnp.inf) | (terms == jnp.inf)
    nonnan = jnp.sum(~mask, axis=0, dtype=jnp.float32)
    # terms = jnp.where(mask, 0.``, terms)
    terms = jnp.where(mask, 0., terms)
    jax.debug.print("terms: {}", terms)
    jax.debug.print("mask: {}", mask)
    # breakpoint()
    # def assert_fn():
    #     if jnp.any(mask):
    #         raise AssertionError

    # if jnp.any(mask): raise AssertionError
    # print(jnp.any(mask))
    # jax.debug.print("mask vals: {}", jnp.any(mask))
    # jax.lax.select(jnp.any(mask), assert_fn, lambda: None)
    # loss = jnp.sum(terms, axis=0) / nonnan
    loss = terms / nonnan
    agg_loss = jnp.sum(loss)
    return agg_loss, loss


@partial(jax.jit, static_argnums=[2,3])
def lfi_pce_eig_fori(params: hk.Params, prng_key: PRNGKey, N: int=100, M: int=10, **kwargs):
    """
    Calculates PCE loss using jax.lax.fori_loop to accelerate. Slightly slower than scan.
    More readable than scan.
    TODO: refactor arguments.
    """
    def compute_marginal(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp):
        def loop_body_fun2(i, carry):
            contrastive_lps = carry
            theta, _ = sim_linear_prior(num_samples, keys[i + 1])
            contrastive_lp = log_prob.apply(flow_params, x, theta, xi_broadcast)
            contrastive_lps += jnp.exp(contrastive_lp)
            return contrastive_lps
        conditional_lps = jax.lax.fori_loop(0, M, loop_body_fun2, conditional_lp)
        return jnp.log(conditional_lps)
    
    keys = jrandom.split(prng_key, 3 + M)
    xi = params['xi']
    flow_params = {k: v for k, v in params.items() if k != 'xi'}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])
    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    # conditional_lp could be the initial starting state that is added upon... 
    marginal_lp = compute_marginal(
        M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp
        ) - jnp.log(M+1)

    return - sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)


@partial(jax.jit, static_argnums=[3,5,6])
def lfi_pce_eig_scan(flow_params: hk.Params, xi_params: hk.Params, prng_key: PRNGKey, log_prob_fun: Callable, designs: Array, N: int=100, M: int=10,):
    """
    Calculates PCE loss using jax.lax.scan to accelerate.

    Modified with a lagrangian loss term that shifts from focusing on training the flow
    to optimizing the design.
    """
    def compute_marginal_lp(keys, log_prob_fun, M, N, x, conditional_lp):
        def scan_fun(contrastive_lps, i):
            theta, _ = sim_linear_prior(N, keys[i + 1])
            contrastive_lp = log_prob_fun(flow_params, x, theta, xi)
            contrastive_lps += jnp.exp(contrastive_lp)
            return contrastive_lps, i + 1
        result = jax.lax.scan(scan_fun, conditional_lp, jnp.array(range(M)))
        return jnp.log(result[0])

    keys = jrandom.split(prng_key, 1 + M)
    xi = xi_params
    xi = jnp.broadcast_to(xi, (N, xi.shape[-1]))

    # simulate the outcomes before finding their log_probs
    # BUG: Make `designs` more explicit
    x, theta_0, x_noiseless, noise = sim_linear_data_vmap(designs, N, keys[0])

    conditional_lp = log_prob_fun(flow_params, x, theta_0, xi)
    conditional_lp_exp = jnp.exp(conditional_lp)
    marginal_lp = compute_marginal_lp(
        keys[1:M+1], log_prob_fun, M, N, x, conditional_lp_exp
        ) - jnp.log(M+1)
    
    # EIG = jnp.sum(conditional_lp - marginal_lp)
    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    return - EIG, (conditional_lp, theta_0, x_noiseless, noise)


@partial(jax.jit, static_argnums=[2,3])
def lfi_pce_eig_vmap_distrax(params: hk.Params, prng_key: PRNGKey, N: int=100, M: int=10, **kwargs):
    """
    Calculates PCE loss using vmap inherent to `distrax` distributions. May be faster
    than scan on GPUs.
    TODO: refactor arguments.
    """
    keys = jrandom.split(prng_key, 2)
    xi = params['xi']
    flow_params = {k: v for k, v in params.items() if k != 'xi'}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])

    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    # TODO: Make function that returns M x num_samples priors
    thetas, log_probs = sim_linear_prior_M_samples(num_samples=num_samples, M=M, key=keys[1])
    
    # conditional_lp could be the initial starting state that is added upon... 
    contrastive_lps = jax.vmap(lambda theta: log_prob.apply(params, x, theta, xi_broadcast))(thetas)
    marginal_log_prbs = jnp.concatenate((jax_lexpand(conditional_lp, 1), jnp.array(contrastive_lps)))
    marginal_lp = jax.nn.logsumexp(marginal_log_prbs, 0) - math.log(M + 1)
    # marginal_lp = compute_marginal_lp3(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp)

    return - sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)


@partial(jax.jit, static_argnums=[2,3])
def lfi_pce_eig_vmap_manual(params: hk.Params, prng_key: PRNGKey, N: int=100, M: int=10, **kwargs):
    """
    Calculates PCE loss using explicit vmap of `distrax` distributions. May potentially
    be more stable than using `ditrax` implicit version as of 2/9/23. May be faster
    than scan on GPUs.
    TODO: refactor arguments.
    """
    keys = jrandom.split(prng_key, M + 1)
    xi = params['xi']
    flow_params = {k: v for k, v in params.items() if k != 'xi'}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])

    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    thetas, log_probs = jax.vmap(partial(sim_linear_prior, num_samples))(keys[1:M+1])
    
    # conditional_lp could be the initial starting state that is added upon... 
    contrastive_lps = jax.vmap(lambda theta: log_prob.apply(params, x, theta, xi_broadcast))(thetas)
    marginal_log_prbs = jnp.concatenate((jax_lexpand(conditional_lp, 1), jnp.array(contrastive_lps)))
    marginal_lp = jax.nn.logsumexp(marginal_log_prbs, 0) - math.log(M + 1)
    # marginal_lp = compute_marginal_lp3(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp)

    return - sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)

