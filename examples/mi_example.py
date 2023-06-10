from lfiax.utils.simulators import sim_linear_data_vmap, sim_linear_data_vmap_theta
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from jax import vmap
from jax.scipy.stats import gaussian_kde
from functools import partial
import distrax
import haiku as hk

def make_lin_reg_prior():
    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**1) * jnp.ones(theta_shape)

    prior = distrax.Independent(
        distrax.MultivariateNormalDiag(mu, sigma)
    )
    return prior

# Constants
N = 5000
M = 500
KDE_SAMPLES = 1000
D = 2  # Number of independent measurements
SEED = 420

# Generate d* values
key = jrandom.PRNGKey(SEED)
key, subkey = jrandom.split(key)
d_star = jrandom.normal(subkey, (D, 1))
# d_star = jrandom.uniform(subkey, shape=(1,D), minval=-10, maxval=10)
# d_star = jrandom.normal(subkey, (D,))
# d_star = jnp.array([[7.]])

# Generate θ(i) and θ(s) samples
key, subkey = jrandom.split(key)
num_samples = N + M

# Making more robust to different design sizes
prior = make_lin_reg_prior()

# Evaluate the log-prior for all prior samples
prng_seq = hk.PRNGSequence(SEED)
priors, prior_log_prob = prior.sample_and_log_prob(seed=next(prng_seq), sample_shape=(num_samples,))

# Split priors into θ(i) and θ(s)
theta_i = priors[:N]
theta_s = priors[N:]

# y_noised, priors, y, sigma = sim_linear_data_vmap(d_star, num_samples, subkey)
y_noised, _, _ = sim_linear_data_vmap_theta(d_star.T, theta_i, next(prng_seq))

# Generate ε and ν samples for KDE
key, subkey = jrandom.split(key)
epsilon_samples = jrandom.normal(subkey, (KDE_SAMPLES,1))
key, subkey = jrandom.split(key)
nu_samples = distrax.Gamma(2.0, 0.5).sample(seed=subkey, sample_shape=(KDE_SAMPLES,1))

# Compute custom KDE using JAX primitives
p_noise_samples = epsilon_samples + nu_samples

# Compute KDE using JAX's built-in function
kde = gaussian_kde(p_noise_samples.T)

@jax.jit
def p_yj_given_dj_theta(yj, dj, theta0, theta1):
    return kde.logpdf(yj - (theta1 + theta0 * dj))

def single_sample_log_ratio(y_sample, d_star, theta0_i, theta1_i, theta0_s, theta1_s):
    kde_log_probs_d_star = vmap(p_yj_given_dj_theta, (0, 0, None, None))
    num_val = jnp.sum(kde_log_probs_d_star(y_sample, d_star, theta0_i, theta1_i))

    kde_log_probs_d_star_theta_s = vmap(kde_log_probs_d_star, (None, None, 0, 0))
    log_probs_matrix = kde_log_probs_d_star_theta_s(y_sample, d_star, theta0_s, theta1_s)
    log_prob_array = jnp.sum(log_probs_matrix, axis=1)
    denom_val = jax.scipy.special.logsumexp(log_prob_array) - jnp.log(M)
    
    return num_val - denom_val

# y_noised = y_noised.squeeze(0)[:N]
if D == 1:
    y_noised = y_noised[:N].squeeze(0)

# Define vmap over theta_i and y_noised
vmap_log_ratio = vmap(single_sample_log_ratio, in_axes=(0, None, 0, 0, None, None))

# Call the function like so:
log_ratios = vmap_log_ratio(y_noised, d_star, theta_i[:, 0], theta_i[:, 1], theta_s[:,0], theta_s[:,1])

mi_approximation = jnp.mean(log_ratios)

print("Mutual information approximation:", mi_approximation)
