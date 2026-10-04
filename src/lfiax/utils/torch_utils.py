import os
import argparse
import sys
import time
import random
import numpy as np

import torch
import torchsde

# needed for torchsde
sys.setrecursionlimit(1500)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SIR_SDE(torch.nn.Module):
    '''Taken directly from code from Ivanova et al. 2021 iDAD paper.'''
    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, population_size):

        super().__init__()
        # parameters: (beta, gamma)
        self.params = params
        self.N = population_size

    # For efficiency: implement drift and diffusion together
    def f_and_g(self, t, x):
        with torch.no_grad():
            x.clamp_(0.0, self.N)

        p_inf = self.params[:, 0] * x.prod(-1) / self.N
        p_inf_sqrt = torch.sqrt(p_inf)
        p_rec = self.params[:, 1] * x[:, 1]

        f_term = torch.stack([-p_inf, p_inf - p_rec], dim=-1)
        g_term = torch.stack(
            [
                torch.stack([-p_inf_sqrt, p_inf_sqrt], dim=-1),
                torch.stack([torch.zeros_like(p_rec), -torch.sqrt(p_rec)], dim=-1),
            ],
            dim=-1,
        )
        return f_term, g_term


def solve_sir_sdes(
    num_samples,
    device,
    grid=10000,
    savegrad=False,
    save=False,
    filename="sir_sde_data.pt",
    theta_loc=None,
    theta_covmat=None,
    params=None,
    params_log_probs=None,
    seed=None
):
    '''Taken directly from code from Ivanova et al. 2021 iDAD paper.'''
    set_seed(0 if seed is None else int(seed[0]))
    ####### Change priors here ######
    if params is not None:
        assert params.shape[0] == num_samples, f"The batch dimension (0th) must be of size {num_samples}."
        assert params.shape[-1] == 2, "The last dimension of the tensor must be of size 2."
    else:
        # assert False "drawing torch priors in SDE, don't do that."
        print("wrong priors")
        if theta_loc is None or theta_covmat is None:
            theta_loc = torch.tensor([0.5, 0.1], device=device).log()
            theta_covmat = torch.eye(2, device=device) * 0.5 ** 2

        prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)
        params = prior.sample(torch.Size([num_samples])).exp()
    #################################

    T0, T = 0.0, 100.0  # initial and final time
    GRID = grid  # time-grid

    population_size = 500.0
    initial_infected = 2.0  # initial number of infected

    ## [non-infected, infected]
    y0 = torch.tensor(
        num_samples * [[population_size - initial_infected, initial_infected]],
        device=device,
    )  # starting point
    ts = torch.linspace(T0, T, GRID, device=device)  # time grid

    sde = SIR_SDE(
        population_size=torch.tensor(population_size, device=device), params=params,
    ).to(device)

    start_time = time.time()
    ys = torchsde.sdeint(sde, y0, ts)  # solved sde
    end_time = time.time()
    # return ys0, ys1
    print("Simulation Time: %s seconds" % (end_time - start_time))

    save_dict = dict()
    idx_good = torch.where(ys[:, :, 1].mean(0) >= 1)[0]

    save_dict["prior_samples"] = params[idx_good].cpu()
    save_dict["prior_log_probs"] = (
        params_log_probs[idx_good].cpu() if params_log_probs is not None
        else torch.zeros(len(idx_good))
    )
    save_dict["ts"] = ts.cpu()
    save_dict["dt"] = (ts[1] - ts[0]).cpu()  # delta-t (time grid)
    # drop 0 as it's not used (saves space)
    save_dict["ys"] = ys[:, idx_good, 1].cpu()

    # grads can be calculated in backward pass (saves space)
    if savegrad:
        # central difference
        grads = (ys[2:, ...] - ys[:-2, ...]) / (2 * save_dict["dt"])
        save_dict["grads"] = grads[:, idx_good, :].cpu()

    # meta data
    save_dict["N"] = population_size
    save_dict["I0"] = initial_infected
    save_dict["num_samples"] = save_dict["prior_samples"].shape[0]

    if save:
        print("Saving data.", end=" ")
        torch.save(save_dict, f"sde_data/{filename}")

    print("DONE.")
    return save_dict
