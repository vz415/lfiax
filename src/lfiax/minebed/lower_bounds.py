import numpy as np

# PyTorch stuff
import torch

# ------ MINE-F LOSS FUNCTION ------ #


def minef_loss(x_sample, y_sample, model, device):

    # Shuffle y-data for the second expectation
    idxs = np.random.choice(range(len(y_sample)), size=len(y_sample), replace=False)
    # We need y_shuffle attached to the design d
    y_shuffle = y_sample[idxs]

    # Get predictions from network
    pred_joint = model(x_sample, y_sample)
    pred_marg = model(x_sample, y_shuffle)

    # Compute the MINE-f (or NWJ) lower bound
    Z = torch.tensor(np.exp(1), device=device, dtype=torch.float)
    mi_ma = torch.mean(pred_joint) - torch.mean(
        torch.exp(pred_marg) / Z + torch.log(Z) - 1
    )

    # we want to maximize the lower bound; PyTorch minimizes
    loss = -mi_ma

    return loss


def minef_gradients(x_sample, y_sample, ygrads, model, device):

    # obtain marginal data and log-likelihood gradients
    idx = np.random.permutation(len(y_sample))
    y_shuffle = y_sample[idx]
    ygrads_shuffle = ygrads[idx]

    # Need to create new tensors for the autograd computation to work;
    # This is because y is not a leaf variable in the computation graph
    x_sample = torch.tensor(
        x_sample, dtype=torch.float, device=device, requires_grad=True
    )
    y_sample = torch.tensor(
        y_sample, dtype=torch.float, device=device, requires_grad=True
    )
    y_shuffle = torch.tensor(
        y_shuffle, dtype=torch.float, device=device, requires_grad=True
    )

    # Get predictions from network
    pred_joint = model(x_sample, y_sample)
    pred_marg = model(x_sample, y_shuffle)

    # Compute gradients of lower bound with respect to data y
    dIdy_joint = torch.autograd.grad(
        pred_joint.sum(), (x_sample, y_sample), retain_graph=True
    )[1].data
    dIdy_marg = torch.autograd.grad(
        pred_marg.sum(), (x_sample, y_shuffle), retain_graph=True
    )[1].data

    # Compute gradient through forward differentiation
    dE1 = torch.mean(dIdy_joint * ygrads, axis=0)
    Z = torch.tensor(np.exp(1), device=device, dtype=torch.float)
    dE2 = torch.mean(dIdy_marg * ygrads_shuffle * torch.exp(pred_marg) / Z, axis=0)

    dI = dE1.reshape(-1, 1) - dE2.reshape(-1, 1)

    # we want to maximize the lower bound; PyTorch minimizes
    loss_gradients = -dI

    return loss_gradients
