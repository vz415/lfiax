# SIR with EPIG sampling

This is an experimental port of the SIR BOED loop from the private experiment
branch (`68d763c`). It includes the corrected scalar neural spline flow, continuous
SIR trajectory interpolation, and EPIG-prioritized positive sampling. It is an
updated experiment, not a claim to reproduce the older paper results. The scalar
flow correction changes optimization dynamics; EPIG is enabled in the supplied
configuration. The regular design-distribution variant remains available with
`epig.enabled=false`, but is not the recommended configuration for this port.

## Install

Use Python 3.10 or 3.11 in a fresh environment and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[sir]'
```

The experiment uses JAX for inference/design and PyTorch plus TorchSDE for SIR
trajectories. CPU is the default. For GPU execution, install JAX and PyTorch builds
appropriate for your CUDA environment, then set `experiment.device=cuda`.
`experiment.device` selects the Torch simulator device; JAX selects its device
through its own installation/environment settings.

## Generate and load observations

The simulator comes from Ivanova et al.'s
[iDAD SIR experiment](https://github.com/desi-ivanova/idad#sir-experiment), whose
README describes TorchSDE installation and generating training/test trajectories
with `epidemic_simulate_data.py`. LFiax includes its adapted SIR simulator, so no
separate iDAD installation is needed:

```bash
python sir_simulate_true_data.py --device cpu --seed 0
```

This generates `sde_data/sir_sde_data_real_8_0.pt`, using beta=0.8, gamma=0.1,
population=500, initial infected=2, times 0–100 and a 10,000-point observation grid.
The Torch dictionary contains `ys` (time × surviving trajectories), `ts`, `dt`,
`prior_samples`, `prior_log_probs`, `N`, `I0`, and `num_samples`. Extinct trajectories
are filtered by the inherited simulator. New stochastic observations will not be
identical to the historical dataset unless the original random state is available.

For an existing compatible iDAD/LFiax dataset, skip generation and pass its path
as `data.observations=/absolute/path/to/observations.pt`. The default setup uses
R0=8 observations; the legacy `experiment.sir_type` field does not choose a dataset.
Only load trusted Torch/pickle files.

Training and posterior-predictive trajectories are generated online on the
100,000-point training grid. A precomputed prior pickle is **not required**.
Legacy debug mode optionally accepts `experiment.debug=true data.prior_pool=...`;
that pickle must contain `ts` as a CPU Torch tensor, `final_ys` with shape
(time, samples), and paired `theta_0` with shape (samples, 2).

## Run

```bash
python sir.py
```

Hydra accepts `key=value` overrides. The default is local CPU execution with W&B
disabled, two design rounds, N=256, M=255, a 1,000-pair posterior pool, and EPIG
selecting 256 positives from 512 candidates using eight dropout particles.
EPIG uses `top_k` selection and shared dropout randomness by default. The flow
uses dropout=0.2 and spline bounds [-20, 7].

For an exploratory shorter run (still generates SDE trajectories and posterior
inference/diagnostics):

```bash
python sir.py experiment.design_rounds=1 \
  optimization_params.training_steps=20 \
  optimization_params.refine_likelihood_rounds=1 \
  contrastive_sampling.N=16 contrastive_sampling.M=15 \
  epig.num_candidates=32 epig.num_particles=2 \
  experiment.posterior_pool_size=64 \
  mcmc_params.num_adapt_steps=100 mcmc_params.num_mcmc_samples=100
```

EPIG requires positive dropout, M < N, and N <= candidates <= posterior pool size.
The loop includes posterior refinement and LC2ST diagnostics, so even a short
training run can take time. Outputs are saved under `sir/eig_lambda_.../`; Hydra
also writes logs under `data/`. Generated data and outputs are ignored by Git.
To log to your W&B account, set `wandb.use_wandb=true wandb.entity=YOUR_ENTITY`.

This release is intended for exploration. It has basic port checks rather than
full experiment/result validation.
