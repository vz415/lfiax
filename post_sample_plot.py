import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
import numpy as np
import distrax
import haiku as hk
from lfiax.flows.nsf import make_nsf
from lfiax.utils.simulators import sim_linear_data_vmap_theta
from lfiax.utils.utils import plot_prior_posterior, save_posterior_marginal

import pickle as pkl
import matplotlib.pyplot as plt

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


# Unpickling the weights and optimizer state to check things out
# Load the objects.
with open("SNPE_best_1D_params_V2.pkl", "rb") as f:
    loaded_objects = pkl.load(f)

# Retrieve the loaded objects.
post_params = loaded_objects['flow_params']
xi_params = loaded_objects['xi_params']

hidden_size = 128
mlp_num_layers = 4 # default is 4
num_bins = 4 # Default is 4
flow_num_layers = 1 # CHANGED from default of 5
theta_shape = (2,)
EVENT_SHAPE = (xi_params['xi'].shape[0],)
xi_shape = (xi_params['xi'].shape[0],)


def make_lin_reg_prior():
    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    prior = distrax.Independent(
        distrax.MultivariateNormalDiag(mu, sigma)
    )
    return prior

# TODO: test out with some different seeds
prng_seq = hk.PRNGSequence(10)
priors = make_lin_reg_prior()
prior_samples = priors.sample(seed=next(prng_seq), sample_shape=(1_000,))

prior_shift = jnp.mean(prior_samples, axis=0)
prior_scale = jnp.std(prior_samples, axis=0)

x_mean_shift, _, _ = sim_linear_data_vmap_theta(xi_params['xi'], prior_samples, next(prng_seq))

x_mean_norm1 = jnp.mean(x_mean_shift, axis=0)
x_scale_norm1 = jnp.std(x_mean_shift, axis=0)

true_theta = jnp.array([[5,2]])
d_sim = xi_params['xi']
sim_samples = 10_000

prior_samples = priors.sample(seed=next(prng_seq), sample_shape=(sim_samples,))
x_obs, _, _ = sim_linear_data_vmap_theta(d_sim, true_theta, next(prng_seq))

x_obs_scale = (x_obs - x_mean_norm1) / x_scale_norm1

@jax.jit
def inverse_standard_scale(scaled_x, shift, scale):
    return (scaled_x * scale) + shift

@hk.without_apply_rng
@hk.transform
def post_sample(key: PRNGKey, num_samples: int,
                shift: Array, scale: Array,
                x: Array, xi: Array) -> Array:
    # Does sampling the likelihood require x?
    """vi is sampling the posterior distribution so doesn't need
    conditional information. Just uses distrax bijector layers.
    """
    model = make_nsf(
        event_shape=theta_shape,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_theta=False, # Depends on a batch so must do manually
        use_resnet=True,
        conditional=True
    )
    
    # TODO: test this for multiple values of X
    x = jnp.broadcast_to(x, (num_samples, x.shape[-1]))
    xi = jnp.broadcast_to(xi, (num_samples, xi.shape[-1]))

    samples = model._sample_n(key=key, 
                            n=num_samples,
                            theta=x,
                            xi=xi,
                            )
    
    return inverse_standard_scale(samples, shift, scale)
    # return samples


post_samples = post_sample.apply(post_params, key=next(prng_seq),
                                 num_samples=sim_samples,
                                 shift=prior_shift, scale=prior_scale,
                                 x=x_obs_scale,
                                 xi=xi_params['xi'][None,:]/10.)

# breakpoint()
plot_prior_posterior(prior_samples, post_samples, true_theta, 'test_snpe_sampling.png')

print(jnp.mean(post_samples, axis=0))
print(jnp.median(post_samples, axis=0))

save_posterior_marginal(post_samples[:,0], "theta_1.png")
save_posterior_marginal(post_samples[:,1], "theta_0.png")

# # Initialize the KDE estimator
# kde_estimator = gaussian_kde(post_samples[:,0].reshape(1, -1), bw_method=0.5)

# # Evaluate the KDE on a range of values
# x_vals = jnp.linspace(jnp.min(post_samples[:,0]), jnp.max(post_samples[:,0]), 1000).reshape(1, -1)
# kde_vals = kde_estimator.pdf(x_vals)

# # Plot the resulting KDE
# plt.figure(figsize=(10, 6))
# plt.plot(np.array(x_vals.squeeze()), np.array(kde_vals), label='KDE')
# plt.hist(np.array(post_samples[:,0]), bins=30, density=True, alpha=0.5, label='Histogram')
# plt.xlim(-10,10)
# plt.legend()
# plt.savefig('smoothed.png', dpi=300, bbox_inches='tight')
