import os, sys
from functools import partial
import jax
import jax.numpy as jnp

import haiku as hk

from lfiax.utils.oed import sdm_minebed

import torch
from torch.distributions import Uniform, TransformedDistribution, Distribution
from torch.distributions.transforms import Transform
from torch.distributions import constraints

import distrax

sys.path.append(os.path.join(os.path.dirname(__file__), 'bmp_simulator'))

from simulate_bmp import bmp_simulator


class LogUniform(Transform):
    """
    Defines a transformation for a log-uniform distribution.
    """
    bijective = True
    sign = +1  # Change to -1 if the transform is decreasing in the interval

    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def _call(self, x):
        return torch.exp(x * (self.high - self.low) + self.low)

    def _inverse(self, y):
        return (torch.log(y) - self.low) / (self.high - self.low)

    def log_abs_det_jacobian(self, x, y):
        return (self.high - self.low) * x + self.low

    @property
    def domain(self):
        return constraints.interval(0.0, 1.0)

    @property
    def codomain(self):
        return constraints.positive


def make_torch_bmp_prior():
    low = torch.log(torch.tensor(1e-6))
    high = torch.log(torch.tensor(1.0))

    uniform = Uniform(low=torch.tensor(1e-6), high=torch.tensor(1.0))

    log_uniform = TransformedDistribution(uniform, LogUniform(low, high))

    return log_uniform


class MultiLogUniform(Distribution):
    """
    A class that represents multiple independent log-uniform distributions.
    """
    def __init__(self, num_priors):
        super().__init__()
        self.priors = [make_torch_bmp_prior() for _ in range(num_priors)]

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([prior.sample(sample_shape) for prior in self.priors], dim=-1)

num_priors = 2
priors = MultiLogUniform(num_priors)

# Simulator (BMP onestep model) to use
model_size = (1,1,1)
fixed_receptor = True
simulator = partial(
    bmp_simulator, 
    model_size=model_size,
    model='onestep', 
    fixed_receptor=fixed_receptor)

y_obs = None
DATASIZE = 5_000
BATCHSIZE = DATASIZE
BO_MAX_NUM = 5
NN_layers = 1
NN_hidden = 150
design_dims = 10

thetas = priors.sample((DATASIZE,)).numpy()

opt_design, sim_samples = sdm_minebed(
    simulator = simulator,
    params = priors,
    y_obs = y_obs,
    DATASIZE = DATASIZE,
    BATCHSIZE = BATCHSIZE,
    BO_MAX_NUM = BO_MAX_NUM, 
    dims = design_dims,
    NN_layers = NN_layers,
    NN_hidden = NN_hidden,
    prior_sims = thetas,
    )