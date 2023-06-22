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

## Contributing

Contributions are welcome. Please open issues if anything isn't working as expected or you would like to see new/expanded features.