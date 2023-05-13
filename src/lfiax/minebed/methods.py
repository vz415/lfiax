import itertools
import numpy as np

from GPyOpt.optimization.optimizer import OptLbfgs
from GPyOpt.core.task.space import Design_space

import torch
from torch.autograd import Variable


def indicator_boundaries(bounds, d):
    """
    Checks if the provided design 'd' is within the specified 'bounds'.

    Parameters
    ----------
    bounds: np.ndarray
        Bounds of the design domain.
    d: np.ndarray
        Proposed design.
    """

    bounds = np.array(bounds)
    check = bounds.T - d
    low = all(i <= 0 for i in check[0])
    high = all(i >= 0 for i in check[1])

    if low and high:
        ind = 1.
    else:
        ind = 0.

    return np.array([[ind]])


def fun_dfun(obj, space, d):
    """
    Computes the posterior predictive and posterior predictive gradients of the
    provided GPyOpt object.

    Parameters
    ----------
    obj: GPyOpt object
        The GPyOpt object with a surrogate probabilistic model.
    space: GPyOpt space
        A GPyOpt object that contains information about the design domain.
    d: np.ndarray
        Proposed design.
    """

    mask = space.indicator_constraints(d)

    pred = obj.model.predict_withGradients(d)[0][0][0]
    d_pred = obj.model.predict_withGradients(d)[2][0]

    return float(pred * mask), d_pred * mask


def get_GP_optimum(obj):
    """
    Finds the optimal design by maximising the mean of the surrogate
    probabilistic GP model.

    Parameters
    ----------
    obj: GPyOpt object
        The GPyOpt object with a surrogate probabilistic model.
    """

    # Define space
    space = Design_space(obj.domain, obj.constraints)
    bounds = space.get_bounds()

    # Get function to optimize + gradients
    # Also mask by everything that is allowed by the constraints
    # fun = lambda d: fun_dfun(obj, space, d)[0]
    # f_df = lambda d: fun_dfun(obj, space, d)
    # def fun(d):
    #    return fun_dfun(obj, space, d)[0]
    # Specify Optimizer --- L-BFGS
    optimizer = OptLbfgs(space.get_bounds(), maxiter=1000)

    # Do the optimisation
    x, _ = optimizer.optimize(
        x0=obj.x_opt,
        f=lambda d: fun_dfun(obj, space, d)[0],
        f_df=lambda d: fun_dfun(obj, space, d))
    # TODO: MULTIPLE RE-STARTS FROM PREVIOUS BEST POINTS

    # Round values if space is discrete
    xtest = space.round_optimum(x)[0]

    if space.indicator_constraints(xtest):
        opt = xtest
    else:
        # Rounding mixed things up, so need to look at neighbours

        # Compute neighbours to optimum
        idx_comb = np.array(
            list(itertools.product([-1, 0, 1], repeat=len(bounds))))
        opt_combs = idx_comb + xtest

        # Evaluate
        GP_evals = []
        combs = []
        for idx, d in enumerate(opt_combs):

            cons_check = space.indicator_constraints(d)[0][0]
            bounds_check = indicator_boundaries(bounds, d)[0][0]

            if cons_check * bounds_check == 1:
                pred = obj.model.predict(d)[0][0][0]
                GP_evals.append(pred)
                combs.append(d)
            else:
                pass

        idx_opt = np.where(GP_evals == np.min(GP_evals))[0][0]
        opt = combs[idx_opt]

    return opt


def compute_weights(model, params, y_obs, LB_type='NWJ'):
    """
    Computes the ratios of posterior to prior distribution for a given trained
    'model' and a set of 'params'.

    Parameters
    ----------
    model: torch.nn.Module (or child class object)
        Neural network trained using a lower bound of the form 'LB_type' as a
        objective function.
    params: np.ndarray of size (:, dim(parameter))
        Set of parameter samples at which to evaluate the posterior to prior
        distribution ratio.
    y_obs: np.darray of size (1, dim(Y))
        Observed data obtained after the BED procedure.
    LB_type: str
        The type of mutual information lower bound that was maximised.
        (default is 'NWJ', also known as MINE-f)
    """

    # Define PyTorch variables
    x = Variable(
        torch.from_numpy(params).type(torch.FloatTensor),
        requires_grad=True)
    y = Variable(
        torch.from_numpy(y_obs).type(torch.FloatTensor),
        requires_grad=True)

    # Pass observed data and parameters through the model
    w = []
    for idx in range(len(x)):
        T = model(x[idx], y).data.numpy()
        if LB_type == 'NWJ':
            w.append(np.exp(T - 1))
        else:
            raise NotImplementedError
    w = np.array(w)

    return w.reshape(-1)


def posterior_samples(
        model, params, y_obs,
        size=1, method='categorical', LB_type='NWJ'):
    """
    Generates samples from the posterior distribution of the model parameters,
    given a real-world observation 'y_obs'.

    Parameters
    ----------
    model: torch.nn.Module (or child class object)
        Neural network trained using a lower bound of the form 'LB_type' as a
        objective function.
    params: np.ndarray of size (:, dim(parameter))
        Set of prior parameter samples.
    y_obs: np.darray of size (1, dim(Y))
        Observed data obtained after the BED procedure.
    size: int
        Number of posterior samples to be generated.
        (default is 1)
    method: str
        Method used to generate posterior samples.
        (default is 'categorical')
    LB_type: str
        The type of mutual information lower bound that was maximised.
        (default is 'NWJ', also known as MINE-f)
    """

    # Decide on a method to sample from the posterior
    if method == 'categorical':

        # Compute and normalise weights for categorical sampling
        w = compute_weights(model, params, y_obs, LB_type)
        ws_norm = w.reshape(-1) / np.sum(w.reshape(-1))

        samples = []
        for _ in range(size):
            idx = np.random.choice(range(len(params)), p=ws_norm)
            samples.append(params[idx])
        samples = np.array(samples)

    else:
        raise NotImplementedError
        # TODO: Implement MCMC

    return samples


def posterior_density(
        model, params, params_densities,
        y_obs, LB_type='NWJ'):
    """
    Computes the posterior density the model parameters, given a real-world
    observation 'y_obs'.

    Parameters
    ----------
    model: torch.nn.Module (or child class object)
        Neural network trained using a lower bound of the form 'LB_type' as a
        objective function.
    params: np.ndarray of size (:, dim(parameter))
        Set of prior parameter samples.
    params_densities: np.ndarray of size (len(params), 1)
        Prior densities of 'params'.
    y_obs: np.darray of size (1, dim(Y))
        Observed data obtained after the BED procedure.
    LB_type: str
        The type of mutual information lower bound that was maximised.
        (default is 'NWJ', also known as MINE-f)
    """

    # Compute weights and multiply with prior densities
    w = compute_weights(model, params, y_obs, LB_type)
    density = np.array(w) * params_densities

    return density
