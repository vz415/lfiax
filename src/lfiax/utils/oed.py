# General math and plotting libraries
from typing import Any, Callable, Dict, Optional, Union, AnyStr, List
import numpy as np
from multiprocessing import Process
import warnings
from itertools import chain
import time

# Torch libraries may be needed for conversions btwn minded & sbi
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributions import gamma

# Minebed libraries needed
import lfiax.minebed.networks as mn
import lfiax.minebed.static.bed as bed


# TODO: Make input variables typed
def sdm_minebed(
    simulator: Callable,
    params: List[Optional[Any]],
    y_obs=None,  # TODO: assign type
    DATASIZE: int = 5000,
    BATCHSIZE: int = 5000,
    N_EPOCH: int = 10000,
    BO_INIT_NUM: int = 5,
    BO_MAX_NUM: int = 5,
    dom_min: float = 0.001,
    dom_max: int = 1000,
    dims: int = 1,
    NN_layers: int = 1,
    NN_hidden: int = 50,
    prior_sims: List[Optional[Any]] = None,
):
    """
    Function that implements implicit MINEBED algorithm. Takes samples from
     the prior and returns an optimal experimental design. Optionally, can
     save designs and convergence criteria each round.

    Args:
        simulator: The simulator to use for minebed. Must take a design and numpy sampled parameters in shape (:, len(theta))
        params: torch dist or numpy array of sampled parameters from the prior
            so just take prior and sample here... unless you plot and save conditional distribution, then just pass previously sampled parameters...
        DATASIZE: Number of sampled prior parameters
        BATCHSIZE: Hyperparameter for NN training that will just be DATASIZE bc it's small enough to work.
        N_EPOCH: Number of epochs to train the NN. Set to 10k for simple two-layer NN.
        BO_INIT_NUM: Number of initial BO evaluations used to initialise the GP.
        BO_MAX_NUM: Max number of BO evaluations/acquisitions to apply after the initialization.

    Returns:
        bed_obj.d_opt: array_like, shape(__,)
            Optimal design(s) returned for the given model, simulator, and
            prior distribution (params).
    """
    # ----- PRIOR AND DOMAIN ----- #
    # Params sampled from the prior/posterior
    params = prior_sims

    dom = [
        {
            "name": "var_1",
            "type": "continuous",
            "domain": (dom_min, dom_max),
            "dimensionality": dims,
        }
    ]

    # No constraints, yet
    con = None

    # ---- DEFINE MODEL ----- #
    # var1 is parameter dimensionality &
    # var2 is simulated outputs
    net = mn.FullyConnected(
        var1_dim=params.shape[1], var2_dim=dims, L=NN_layers, H=NN_hidden
    )

    opt = optim.Adam(net.parameters(), lr=1e-3)
    sch = StepLR(opt, step_size=1000, gamma=0.95)

    # ----- TRAIN MODEL ------ #
    bed_obj = bed.GradientFreeBED(
        model=net,
        optimizer=opt,
        scheduler=sch,
        simulator=simulator,
        prior=params,
        domain=dom,
        n_epoch=N_EPOCH,
        batch_size=BATCHSIZE,
        ma_window=100,
        constraints=con,
        y_obs=None,
    )

    bed_obj.train(BO_init_num=BO_INIT_NUM, BO_max_num=BO_MAX_NUM, verbosity=True)
    
    return bed_obj
