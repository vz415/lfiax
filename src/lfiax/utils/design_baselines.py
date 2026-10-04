import math
from typing import Sequence

import numpy as np


BASELINE_DESIGN_POLICIES = ("optimized", "random", "sobol")
BASELINE_LIKELIHOOD_OBJECTIVES = ("infonce_lambda", "nle")


def validate_baseline_config(design_policy: str, likelihood_objective: str) -> None:
    if design_policy not in BASELINE_DESIGN_POLICIES:
        raise ValueError(
            f"Unknown baseline.design_policy '{design_policy}'. "
            f"Expected one of {BASELINE_DESIGN_POLICIES}."
        )
    if likelihood_objective not in BASELINE_LIKELIHOOD_OBJECTIVES:
        raise ValueError(
            f"Unknown baseline.likelihood_objective '{likelihood_objective}'. "
            f"Expected one of {BASELINE_LIKELIHOOD_OBJECTIVES}."
        )


def select_baseline_design(
    design_policy: str,
    design_round: int,
    seed: int,
    low: float,
    high: float,
    shape: Sequence[int] = (1,),
):
    """Select one fixed baseline design for a sequential design round."""
    if design_policy == "optimized":
        raise ValueError("optimized does not select a fixed baseline design.")
    if design_policy not in ("random", "sobol"):
        raise ValueError(f"Unknown baseline design policy '{design_policy}'.")

    shape = tuple(shape)
    flat_dim = int(np.prod(shape))
    if design_policy == "random":
        rng = np.random.default_rng(seed + design_round)
        unit = rng.uniform(size=flat_dim)
    else:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=flat_dim, scramble=True, seed=seed)
        m = 0 if design_round == 0 else math.ceil(math.log2(design_round + 1))
        unit = sampler.random_base2(m=m)[design_round]

    design = low + (high - low) * unit.reshape(shape)
    return np.asarray(design)
