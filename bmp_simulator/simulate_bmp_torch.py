from typing import Any, Callable, Dict, Optional, Union, AnyStr, Tuple
from xmlrpc.client import boolean
import numpy as np
import random, os
import importlib
import multiprocessing
import functools
import itertools
import promisys.bmp_util as psb
from tqdm import tqdm as tqdm


def bmp_simulator(
    d: np.ndarray,
    p: np.ndarray,
    model_size: Tuple[int, int, int] = (1, 1, 1),
    model: str = "onestep",
    fixed_receptor: bool = True,
    n_threads: int = 4,
):
    """
    Purpose is to wrap the promisys simulator and return a numpy
    array compatible with minebed and sbi. Need to split the processing of the
    inputs depending on whether fixed_receptor=True.

    Args:
        d: array_like, shape (num_designs,) or (1,) Design (of ligands) that will be put into the simulator. 
            Assign as class attribute for promisys. 
        p: array_like, shape (2 * n_L * n_A * n_B,) for onestep or 
            ((2 * n_L * n_A * n_B) + n_L * n_A,) for twostep or (len(priors), ) for a custom simulator.
            Numpy array sampled from the distribution object of the priors, either as a 
            Design (of ligands) that will be put into the simulator.
            Assign as class attribute for promisys.
        p: array_like, shape (2 * n_L * n_A * n_B,) for onestep or ((2 * n_L * n_A * n_B) + n_L * n_A,) for twostep or (len(priors), ) for a custom simulator.
            Numpy array sampled from the distribution object of the priors, either as a
            torch distribution or sbi posterior object.
        model_size: tuple of ints, shape (3,) BMP model to simulate for number of unique ligands, type 1, and type 2 receptors.
        mode: Whether to use the 'onestep' or 'twostep' BMP model
        fixed_receptor: Determine whether to use bmp model with known/fixed receptor, or stochastic according to log distribution. (see promisys code)

    Returns:
        S: something
    """
    d = d.T
    # Check that the design is the right size (N, 1) not (1, N)
    n_L, n_A, n_B = model_size
    num_receptors = n_A + n_B
    ligands = d
    # breakpoint()
    # Splitting passed prior's columns to work with promisys
    # TODO: Think about making splitting and error assertion into its own function
    if fixed_receptor:
        if model == "onestep":
            # assert that p passed is correct size for constant receptor
            if p.shape[1] != (n_L * n_A * n_B + n_L * n_A * n_B):
                raise ValueError(
                    "Number of priors is not consistent with model size or fixed_receptor value."
                )

            p = np.hsplit(p, [n_L * n_A * n_B])

        elif model == "twostep":
            if p.shape[1] != (n_L * n_A * n_B + n_L * n_A + n_L * n_A * n_B):
                raise ValueError(
                    "Number of priors is not consistent with model size or fixed_receptor value."
                )

            p = np.hsplit(p, [n_L * n_A * n_B + n_L * n_A])

        # ----- multiprocessing starmap ------
        # Multiprocess the simulations
        Rs = None
        n_threads = n_threads
        with multiprocessing.Pool(n_threads) as pools:
            # %time
            S = pools.starmap(
                functools.partial(
                    psb.sim_S_LAB,
                    model_size,
                    ligands,
                    Rs,
                    fixed_receptor=fixed_receptor,
                    model=model,
                ),
                zip(*p),
            )

    else:
        if model == "onestep":
            # assert that p passed is correct size for constant receptor
            if p.shape[1] != (n_A + n_B + n_L * n_A * n_B + n_L * n_A * n_B):
                raise ValueError(
                    "Number of priors is not consistent with model size or fixed_receptor value."
                )

            p = np.hsplit(p, [num_receptors, num_receptors + n_L * n_A * n_B])

        elif model == "twostep":
            if p.shape[1] != (
                n_A + n_B + n_L * n_A * n_B + n_L * n_A + n_L * n_A * n_B
            ):
                raise ValueError(
                    "Number of priors is not consistent with model size or fixed_receptor value."
                )

            p = np.hsplit(
                p, [num_receptors, num_receptors + n_L * n_A * n_B + n_L * n_A]
            )

        # ----- multiprocessing starmap ------
        # Multiprocess the simulations
        n_threads = n_threads
        with multiprocessing.Pool(n_threads) as pools:
            # %time
            S = pools.starmap(
                functools.partial(
                    psb.sim_S_LAB,
                    model_size,
                    ligands,
                    fixed_receptor=fixed_receptor,
                    model=model,
                ),
                zip(*p),
            )

    # Turn promisys output into numpy arrays
    S = np.asarray(S, dtype=np.float32).squeeze(axis=1)

    # Temporary fix for nL=1 condition & shapes with extra dimensions
    if len(S.shape) > 2:
        S = S.squeeze(axis=-1)

    return S
