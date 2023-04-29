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
N = 10
M = 10
KDE_SAMPLES = 50000
D = 10  # Number of independent measurements
SEED = 420

# Generate d* values
key = jrandom.PRNGKey(SEED)
key, subkey = jrandom.split(key)
# d_star = jrandom.normal(subkey, (D, 1))
d_star = jrandom.uniform(subkey, shape=(1,D), minval=-10, maxval=10)
# d_star = jrandom.normal(subkey, (D,))
# d_star = jnp.array([[10.]])

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
y_noised, _, _ = sim_linear_data_vmap_theta(d_star, theta_i, next(prng_seq))


# Generate ε and ν samples for KDE
key, subkey = jrandom.split(key)
epsilon_samples = jrandom.normal(subkey, (KDE_SAMPLES,D))
key, subkey = jrandom.split(key)
nu_samples = distrax.Gamma(2.0, 0.5).sample(seed=subkey, sample_shape=(KDE_SAMPLES,D))

# Compute custom KDE using JAX primitives
p_noise_samples = epsilon_samples + nu_samples

# Compute KDE using JAX's built-in function
kde = gaussian_kde(p_noise_samples)

@jax.jit
def p_yj_given_dj_theta_vmap(yj, dj, theta0, theta1):
    # breakpoint()
    has_batch_dim = False
    if hasattr(theta0, 'batch_dim'):
        # has_batch_dim = True
        return kde.logpdf(yj - (theta1 + theta0 * dj))
    # if has_batch_dim:
    # breakpoint()
    return kde.logpdf(yj.T - (theta1[:,None] + theta0[:,None] * dj))
        # theta0 = theta0.val
        # theta1 = theta1.val
    
    # return kde.logpdf(yj - (theta1[:,None] + theta0[:,None] * dj))
    # return kde.logpdf(yj - (theta1 + theta0 * dj))


def single_sample_log_ratio(y_sample, d_star, theta_i_single, theta_s, M):
    log_p_y_i_theta_i = jnp.sum(
        p_yj_given_dj_theta_vmap(y_sample, d_star, theta_i_single[0], theta_i_single[1])
    )
    # breakpoint()
    log_p_y_i_theta_s = jax.scipy.special.logsumexp(
        p_yj_given_dj_theta_vmap(
            y_sample[:, None], d_star, theta_s[:, 0], theta_s[:, 1]
            # y_sample, d_star, theta_s[:, 0], theta_s[:, 1]
        ),
        axis=0,
    ) - jnp.log(M)
    return log_p_y_i_theta_i - log_p_y_i_theta_s

# print(y_noised.shape)
# y_noised = y_noised.squeeze(0)[:N]
y_noised = y_noised[:N]

vmap_log_ratio_theta_i = vmap(single_sample_log_ratio, (None, None, 0, None, None))

log_ratios = vmap_log_ratio_theta_i(y_noised[0], d_star, theta_i, theta_s, M)
ratios = jnp.exp(log_ratios)
mi_approximation = jnp.mean(ratios)
print("Mutual information approximation:", mi_approximation)
