<div align="center">
<img src="docs/_static/scaled_lfiax.png" width="350" alt="logo"/>
</div>

# LFIAX
_Likelihood Free Inference in jAX. Fiat Lux._

JAX-based package for conditional density estimation using normalizing flows. This package essentially wraps the normalizing flow example from the [distrax](https://github.com/deepmind/distrax) repo, as well as modifies the prerequisite bijective functions and distribution objects. The directory structure is divided into the customized bijectors, distributions, conditional neural networks (nets), utility functions, and misc. files from my research. Some misc. files include objective function for training amortized likelihoods or posteriors in the Simulation-based inference (AKA Likelihood Free Inference) framework as well as Bayesian optimal experimental design (BOED) objectives. 

Examples show how this repo can be used for conditional image generation via MNIST and to perform BOED and inference on a simple linear regression model.

## Installation

To install, in the parent directory simply run:

```bash
pip install -e .
```

## API

We show how to use the `log_prob` and `sample` methods from lfiax on MNIST density estimation and sampling.Notice that the base distirbution is uniform and that the wrapped distrax model distribution takes the data x, parameters theta, and conditional designs xi. We can ignore the xi argument and the network will still work.

```python
@hk.without_apply_rng
@hk.transform
def log_prob(x: Array, theta: Array, xi: Array) -> Array:
    """MNIST log-probbility function that uses the `make_nsf` flow generator.
    """
    model = make_nsf(
        event_shape=MNIST_IMAGE_SHAPE,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_theta=True,
        use_resnet=True,
        conditional=True,
        base_dist="uniform",
    )

    return model.log_prob(x, theta, xi)
```

To sample from the learned MNIST density, use the following function. It uses the same function but samples and places a dummy variable for the unused xi argument.

```python
@hk.without_apply_rng
@hk.transform
def model_sample(key: PRNGKey, num_samples: int, cond_data: Array) -> Array:
    model = make_nsf(
        event_shape=MNIST_IMAGE_SHAPE,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_theta=False,
        base_dist="uniform",
    )
    z = jnp.repeat(cond_data, num_samples, axis=0)
    z = jnp.expand_dims(z, -1)
    dummy_xi = jnp.array([])
    dummy_xi = jnp.broadcast_to(dummy_xi, (num_samples, dummy_xi.shape[-1]))
    return model._sample_n(key=key, n=[num_samples], theta=z, xi=dummy_xi)
```

See the MNIST example file for more details.

## Contributing

Contributions are welcome. Please open issues if anything isn't working as expected or you would like to see new/expanded features.
