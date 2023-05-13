import numpy as np
from tqdm import tqdm as tqdm

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from itertools import chain


class MINE:
    """
    A class used to train a lower bound on the mutual information between two
    random variables.

    Attributes
    ----------
    model: torch.nn.Module (or child class) object
        A parametrised neural network that is trained to compute a lower bound
        on the mutual information between two random variables.
    optimizer: torch.optim object
        The optimiser used to learn the neural network parameters.
    scheduler: torch.optim.lr_scheduler object
        The learning rate scheduler used in conjunction with the optimizer.
    LB_type: str
        The type of mutual information lower bound that is maximised.
    X: np.ndarray
        Numpy array of the first random variable
    Y: np.ndarray
        Numpy array of the second random variable
    train_lb: np.ndarray
        Numpy array of the lower bound evaluations as a function of neural
        network parameter updates during training.

    Methods
    -------
    set_optimizer:
        Set a optimizer to train the neural network (recommended).
    set_scheduler:
        Set a scheduler to update the optimizer (recommended).
    evaluate_model:
        Evaluate the neural network for given two data points.
    evaluate_lower_bound:
        Evaluate the lower bound for two sets of data points.
    train:
        Train the neural network with the mutual information lower bound as
        the objective function to be maximised.
    """

    def __init__(
            self, model, data, LB_type='NWJ',
            lr=1e-3, schedule_step=1e8, schedule_gamma=1,
            ):
        """
        Parameters
        ----------
        model: torch.nn.Module (or child class) object
            A parametrised neural network that is trained to compute a lower
            bound on the mutual information between two random variables.
        data: tuple of np.ndarrays
            Tuple that contains the datasets of the two random variables.
        LB_type: str
            The type of mutual information lower bound that is maximised.
            (default is 'NWJ', also known as MINE-f)
        lr: float
            Learning rate of the Adam optimiser. May ignore if optimizer is
            specified later via the set_optimizer() method.
            (default is 1e-3)
        schedule_step: int
            Step size of the StepLR scheduler. May ignore if scheduler is
            specified later via the set_scheduler() method.
            (default is 1e8, sufficiently large to not be used by default)
        schedule_gamma: float
            Learning rate decay factor (gamma) of the StepLR scheduler. May
            ignore if scheduler is specified later via the set_scheduler()
            method. Should be between 0 and 1.
            (default is 1)
        """

        self.model = model
        self.X, self.Y = data
        self.LB_type = LB_type

        # default optimizer is Adam; may over-write
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        # default scheduler is StepLR; may over-write
        self.scheduler = StepLR(
            self.optimizer, step_size=schedule_step, gamma=schedule_gamma)

    def set_optimizer(self, optimizer):
        """
        Set a custom optimizer to be used during training.

        Parameters
        ----------
        optimizer: torch.optim object
            The custom optimizer object.
        """

        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        """
        Set a custom learning rate scheduler to be used during training.

        Parameters
        ----------
        scheduler: torch.optim.lr_scheduler object
            The custom optimizer object.
        """

        self.scheduler = scheduler

    def _ma(self, a, window=100):
        """Computes the moving average of array a, within a 'window'"""

        avg = [np.mean(a[i:i + window]) for i in range(0, len(a) - window)]
        return avg

    def _lower_bound(self, pj, pm):
        """Evaluates the lower bound with joint/marginal samples."""

        if self.LB_type == 'NWJ':
            # Compute the NWJ bound (also known as MINE-f)
            Z = torch.tensor(np.exp(1))
            lb = torch.mean(pj) - torch.mean(torch.exp(pm) / Z)
        else:
            raise NotImplementedError()

        return lb

    def evaluate_model(self, X, Y):
        """
        Evaluates the current model given two data points/sets 'X' and 'Y'.

        Parameters
        ----------
        X: np.ndarray of shape (:, dim(X))
            Numpy array of samples from the first random variable.
        Y: np.ndarray of shape (:, dim(Y))
            Numpy array of samples from the second random variable.
        """

        # Define PyTorch variables
        x = Variable(
            torch.from_numpy(X).type(torch.FloatTensor),
            requires_grad=True)
        y = Variable(
            torch.from_numpy(Y).type(torch.FloatTensor),
            requires_grad=True)

        # Get predictions from network
        predictions = self.model(x, y)

        return predictions

    def evaluate_lower_bound(self, X, Y):
        """
        Evaluates the lower bound using the current model and samples of the
        first and second random variable.

        Parameters
        ----------
        X: np.ndarray of shape (:, dim(X))
            Numpy array of samples from the first random variable.
        Y: np.ndarray of shape (:, dim(Y))
            Numpy array of samples from the second random variable.
        """

        # shuffle data
        Y_shuffle = np.random.permutation(Y)

        # Get predictions from network
        pred_joint = self.evaluate_model(X, Y)
        pred_marginal = self.evaluate_model(X, Y_shuffle)

        # Compute lower bound
        lb = self._lower_bound(pred_joint, pred_marginal)

        return lb

    def train(self, n_epoch, batch_size=None, bar=True):
        """
        Trains the neural network using samples of the first random variable
        and the second variable. The resulting objective function is stored
        in 'self.train_lb'.

        Parameters
        ----------
        n_epoch: int
            The number of training epochs.
        batch_size: int
            The batch size of data samples used during training.
            (default is None, in which case no batches are used)
        bar: boolean
            Displays a progress bar of the training procedure.
            (default is True)
        """

        # if no batch_size is given, set it to the size of the training set
        if batch_size is None:
            batch_size = len(self.X)

        # start the training procedure
        self.train_lb = []
        for epoch in tqdm(range(n_epoch), leave=True, disable=not bar):

            for b in range(int(len(self.X) / batch_size)):

                # sample batches randomly
                index = np.random.choice(
                    range(len(self.X)), size=batch_size, replace=False)
                x_sample = self.X[index]
                y_sample = self.Y[index]

                # Compute lower bound
                lb = self.evaluate_lower_bound(x_sample, y_sample)

                # maximise lower bound
                loss = - lb

                # save training score
                self.train_lb.append(lb.data.numpy())

                # parameter update steps
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            # scheduler step
            self.scheduler.step()

        self.train_lb = np.array(self.train_lb).reshape(-1)

class BMA_MINE:
    """
    A class used to train a lower bound on the mutual information between two
    random variables.

    Attributes
    ----------
    model: torch.nn.Module (or child class) object
        A parametrised neural network that is trained to compute a lower bound
        on the mutual information between two random variables.
    optimizer: torch.optim object
        The optimiser used to learn the neural network parameters.
    scheduler: torch.optim.lr_scheduler object
        The learning rate scheduler used in conjunction with the optimizer.
    LB_type: str
        The type of mutual information lower bound that is maximised.
    X: np.ndarray
        Numpy array of the first random variable
    Y: np.ndarray
        Numpy array of the second random variable
    train_lb: np.ndarray
        Numpy array of the lower bound evaluations as a function of neural
        network parameter updates during training.

    Methods
    -------
    set_optimizer:
        Set a optimizer to train the neural network (recommended).
    set_scheduler:
        Set a scheduler to update the optimizer (recommended).
    evaluate_model:
        Evaluate the neural network for given two data points.
    evaluate_lower_bound:
        Evaluate the lower bound for two sets of data points.
    train:
        Train the neural network with the mutual information lower bound as
        the objective function to be maximised.
    """

    def __init__(
            self, model1, model2, data, LB_type='NWJ',
            lr=1e-3, schedule_step=1e8, schedule_gamma=1,
            BF1=1., BF2=1.):
        """
        Parameters
        ----------
        model: torch.nn.Module (or child class) object
            A parametrised neural network that is trained to compute a lower
            bound on the mutual information between two random variables.
        data: tuple of np.ndarrays
            Tuple that contains the datasets of the two random variables.
        LB_type: str
            The type of mutual information lower bound that is maximised.
            (default is 'NWJ', also known as MINE-f)
        lr: float
            Learning rate of the Adam optimiser. May ignore if optimizer is
            specified later via the set_optimizer() method.
            (default is 1e-3)
        schedule_step: int
            Step size of the StepLR scheduler. May ignore if scheduler is
            specified later via the set_scheduler() method.
            (default is 1e8, sufficiently large to not be used by default)
        schedule_gamma: float
            Learning rate decay factor (gamma) of the StepLR scheduler. May
            ignore if scheduler is specified later via the set_scheduler()
            method. Should be between 0 and 1.
            (default is 1)
        BF1: float
            Probability of the first model (numerator of the BayesFacor).
        BF2: float
            Probability of the second model (denominator of the BayesFacor).
        """

        self.model1, self.model2 = model1, model2
        self.X1, self.Y1, self.X2, self.Y2 = data
        self.BF1, self.BF2 = BF1, BF2
        # TODO: Make assertions for wrong number of models or data received
        self.LB_type = LB_type

        # Chain together the models to optimize over uniformly
        # all_params = chain(self.model1, self.model2)
        # Nope! Making a set of parameters to optimize instead
        nets = [self.model1, self.model2]
        net_parameters = set()
        for net in nets:
            net_parameters |= set(net.parameters())
        # default optimizer is Adam; may over-write
        self.optimizer = Adam(net_parameters, lr=lr)
        # default scheduler is StepLR; may over-write
        self.scheduler = StepLR(
            self.optimizer, step_size=schedule_step, gamma=schedule_gamma)

    def set_optimizer(self, optimizer):
        """
        Set a custom optimizer to be used during training.

        Parameters
        ----------
        optimizer: torch.optim object
            The custom optimizer object.
        """

        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        """
        Set a custom learning rate scheduler to be used during training.

        Parameters
        ----------
        scheduler: torch.optim.lr_scheduler object
            The custom optimizer object.
        """

        self.scheduler = scheduler

    def _ma(self, a, window=100):
        """Computes the moving average of array a, within a 'window'"""

        avg = [np.mean(a[i:i + window]) for i in range(0, len(a) - window)]
        return avg

    def _lower_bound(self, pj, pm):
        """Evaluates the lower bound with joint/marginal samples."""

        if self.LB_type == 'NWJ':
            # Compute the NWJ bound (also known as MINE-f)
            Z = torch.tensor(np.exp(1))
            lb = torch.mean(pj) - torch.mean(torch.exp(pm) / Z)
        else:
            raise NotImplementedError()

        return lb

    def evaluate_model(self, X, Y, model_number):
        """
        Evaluates the current model given two data points/sets 'X' and 'Y'.

        Parameters
        ----------
        X: np.ndarray of shape (:, dim(X))
            Numpy array of samples from the first random variable.
        Y: np.ndarray of shape (:, dim(Y))
            Numpy array of samples from the second random variable.
        """

        # Define PyTorch variables
        x = Variable(
            torch.from_numpy(X).type(torch.FloatTensor),
            requires_grad=True)
        y = Variable(
            torch.from_numpy(Y).type(torch.FloatTensor),
            requires_grad=True)

        # Get predictions from network
        if model_number == 1:
            predictions = self.model1(x, y)
        elif model_number == 2:
            predictions = self.model2(x, y)

        return predictions

    def evaluate_lower_bound(self, X, Y, model_number):
        """
        Evaluates the lower bound using the current model and samples of the
        first and second random variable.

        Parameters
        ----------
        X: np.ndarray of shape (:, dim(X))
            Numpy array of samples from the first random variable.
        Y: np.ndarray of shape (:, dim(Y))
            Numpy array of samples from the second random variable.
        """

        # shuffle data
        Y_shuffle = np.random.permutation(Y)

        # Get predictions from network
        pred_joint = self.evaluate_model(X, Y, model_number)
        pred_marginal = self.evaluate_model(X, Y_shuffle, model_number)

        # Compute lower bound
        lb = self._lower_bound(pred_joint, pred_marginal)

        return lb

    def train(self, n_epoch, batch_size=None, bar=True):
        """
        Trains the neural network using samples of the first random variable
        and the second variable. The resulting objective function is stored
        in 'self.train_lb'.

        Parameters
        ----------
        n_epoch: int
            The number of training epochs.
        batch_size: int
            The batch size of data samples used during training.
            (default is None, in which case no batches are used)
        bar: boolean
            Displays a progress bar of the training procedure.
            (default is True)
        """

        # if no batch_size is given, set it to the size of the training set
        if batch_size is None:
            batch_size = len(self.X1)

        # start the training procedure
        self.train_lb = []
        for epoch in tqdm(range(n_epoch), leave=True, disable=not bar):

            for b in range(int(len(self.X1) / batch_size)):

                # sample batches randomly
                index = np.random.choice(
                    range(len(self.X1)), size=batch_size, replace=False)
                m1_x_sample = self.X1[index]
                m1_y_sample = self.Y1[index]
                m2_x_sample = self.X2[index]
                m2_y_sample = self.Y2[index]

                # Compute lower bound
                lb_m1 = self.evaluate_lower_bound(m1_x_sample, m1_y_sample, 1)
                lb_m2 = self.evaluate_lower_bound(m2_x_sample, m2_y_sample, 2)
                lb = lb_m1*self.BF1 + lb_m2*self.BF2

                # maximise lower bound
                loss = - lb

                # save training score
                self.train_lb.append(lb.data.numpy())

                # parameter update steps
                self.model1.zero_grad()
                self.model2.zero_grad()
                loss.backward()
                self.optimizer.step()

            # scheduler step
            self.scheduler.step()

        self.train_lb = np.array(self.train_lb).reshape(-1)
