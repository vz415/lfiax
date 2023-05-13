import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    """
    Fully-connected neural network written as a child class of torch.nn.Module,
    used to compute the mutual information between two random variables.

    Attributes
    ----------
    self.fc_var1: torch.nn.Linear object
        Input layer for the first random variable.
    self.fc_var2: torch.nn.Linear object
        Input layer for the second random variable.
    self.layers: torch.nn.ModuleList object
        Object that contains all layers of the neural network.

    Methods
    -------
    forward:
        Forward pass through the fully-connected eural network.
    """

    def __init__(self, var1_dim, var2_dim, L=1, H=10):
        """
        Parameters
        ----------
        var1_dim: int
            Dimensions of the first random variable.
        var2_dim: int
            Dimensions of the second random variable.
        L: int
            Number of hidden layers of the neural network.
            (default is 1)
        H: int or np.ndarray
            Number of hidden units for each hidden layer. If 'H' is an int, all
            layers will have the same size. 'H' can also be an nd.ndarray,
            specifying the sizes of each hidden layer.
            (default is 10)
        """

        super(FullyConnected, self).__init__()

        # check for the correct dimensions
        if isinstance(H, (list, np.ndarray)):
            if len(H) != L:
                raise AssertionError("Incorrect dimensions of hidden units.")
            H = list(map(int, list(H)))
        else:
            H = [int(H) for _ in range(L)]

        # Define layers over your two random variables
        self.fc_var1 = nn.Linear(var1_dim, H[0])
        self.fc_var2 = nn.Linear(var2_dim, H[0])

        # Define any further layers
        self.layers = nn.ModuleList()
        if L == 1:
            fc = nn.Linear(H[0], 1)
            self.layers.append(fc)
        elif L > 1:
            for idx in range(1, L):
                fc = nn.Linear(H[idx - 1], H[idx])
                self.layers.append(fc)
            fc = nn.Linear(H[-1], 1)
            self.layers.append(fc)
        else:
            raise ValueError('Incorrect value for number of layers.')

    def forward(self, var1, var2):
        """
        Forward pass through the neural network.

        Parameters
        ----------
        var1: torch.autograd.Variable
            First random variable.
        var2: torch.autograd.Variable
            Second random variable.
        """

        # Initial layer over random variables
        hidden = F.relu(self.fc_var1(var1) + self.fc_var2(var2))

        # All subsequent layers
        for idx in range(len(self.layers) - 1):
            hidden = F.relu(self.layers[idx](hidden))

        # Output layer
        output = self.layers[-1](hidden)

        return output

class BMA(nn.Module):
    def __init__(
        self, 
        model1var1, model1var2, model2var1, model2var2):
        """
        Connect the MI submodules together and weight by
        Bayesian Model Averaging (BMA).

        Parameters
        ----------
        model1var1: int
            Dimensions of the first random variable for first model.
        model1var2: int
            Dimensions of the second random variable for first model.
        model2var1: int
            Dimensions of the first random variable for second model.
        model2var2: int
            Dimensions of the second random variable for second model.
        L: int
            Number of hidden layers of the neural network.
            (default is 1)
        H: int or np.ndarray
            Number of hidden units for each hidden layer. If 'H' is an int, all
            layers will have the same size. 'H' can also be an nd.ndarray,
            specifying the sizes of each hidden layer.
            (default is 50)
        """
        super().__init__()
        self.nn1 = FullyConnected(
            var1_dim=model1var1, var2_dim=model1var2)
        self.nn2 = FullyConnected(
            var1_dim=model2var1, var2_dim=model2var2)

    def forward(
        self,
        model1var1, model1var2, 
        model2var1, model2var2,
        BF1, BF2):
        """
        Forward pass entire network.

        Parameters
        ----------
        model1var1: torch.autograd.Variable
            First random variable for first model (thetas_{M1}).
        model1var2: torch.autograd.Variable
            Second random variable for first model (y^hat|d, M_2).
        model2var1: torch.autograd.Variable
            First random variable for second model (thetas_{M2}).
        model2var2: torch.autograd.Variable
            Second random variable for second model (y^hat|d, M_1).
        BF1: float/np.ndarray
            Bayes factor for the first model.
        BF2: float/np.ndarray
            Bayes factor for the second model.
        """
        x = F.relu(self.nn1(model1var1, model1var2))
        x = torch.mul(x, BF1)

        y = F.relu(self.nn2(model2var1, model2var2))
        y = torch.mul(y, BF2)

        output = torch.add(x, y)
        return output
