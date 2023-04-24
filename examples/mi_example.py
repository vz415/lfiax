from lfiax.utils.simulators import sim_linear_data_vmap, sim_linear_data_vmap_theta
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from jax import vmap
from jax.scipy.stats import gaussian_kde
from functools import partial
import distrax

# Constants
N = 50_000
M = 5_000
KDE_SAMPLES = 50000
D = 1  # Number of independent measurements
SEED = 420

# Generate d* values
key = jrandom.PRNGKey(SEED)
key, subkey = jrandom.split(key)
# d_star = jrandom.normal(subkey, (D, 1))
# d_star = jrandom.normal(subkey, (D,))
d_star = jnp.array([[10.]])

# Generate θ(i) and θ(s) samples
key, subkey = jrandom.split(key)
num_samples = N + M
y_noised, priors, y, sigma = sim_linear_data_vmap(d_star, num_samples, subkey)

# Split priors into θ(i) and θ(s)
theta_i = priors[:N]
theta_s = priors[N:]

# Generate ε and ν samples for KDE
key, subkey = jrandom.split(key)
epsilon_samples = jrandom.normal(subkey, (KDE_SAMPLES,))
key, subkey = jrandom.split(key)
nu_samples = distrax.Gamma(2.0, 0.5).sample(seed=subkey, sample_shape=(KDE_SAMPLES,))

# Compute custom KDE using JAX primitives
p_noise_samples = epsilon_samples + nu_samples

# Compute KDE using JAX's built-in function
kde = gaussian_kde(p_noise_samples)

@jax.jit
def p_yj_given_dj_theta_vmap(yj, dj, theta0, theta1):
    return kde.logpdf(yj - (theta1 + theta0 * dj))


def single_sample_log_ratio(y_sample, d_star, theta_i_single, theta_s, M):
    log_p_y_i_theta_i = jnp.sum(
        p_yj_given_dj_theta_vmap(y_sample, d_star, theta_i_single[0], theta_i_single[1])
    )
    log_p_y_i_theta_s = jax.scipy.special.logsumexp(
        p_yj_given_dj_theta_vmap(
            y_sample[:, None], d_star, theta_s[:, 0], theta_s[:, 1]
        ),
        axis=0,
    ) - jnp.log(M)
    return log_p_y_i_theta_i - log_p_y_i_theta_s


y_noised = y_noised.squeeze(0)[:N]

vmap_log_ratio_theta_i = vmap(single_sample_log_ratio, (None, None, 0, None, None))

log_ratios = vmap_log_ratio_theta_i(y_noised[0], d_star, theta_i, theta_s, M)
ratios = jnp.exp(log_ratios)
mi_approximation = jnp.mean(ratios)
print("Mutual information approximation:", mi_approximation)
