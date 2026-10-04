import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from collections import deque
from typing import NamedTuple


def add_noise_to_zeros(matrix, noise_level=1e-6):
    """
    Adds noise to zero elements in the matrix.

    Parameters:
    matrix (jnp.ndarray): Input matrix of shape [N, M].
    noise_level (float): Standard deviation of the noise to be added.

    Returns:
    jnp.ndarray: Matrix with noise added to zero elements.
    """

    # Set a random seed for reproducibility
    key = jrandom.PRNGKey(0)

    # Create a mask of zero elements
    zero_mask = matrix == 0

    # Generate noise with the same shape as matrix
    noise = jrandom.normal(key, matrix.shape) * noise_level

    # Add noise to zero elements only
    result = jnp.where(zero_mask, noise, matrix)

    return result

class LossSmoother:
    def __init__(self, beta=0.9, initial_loss=float('inf')):
        self.beta = beta
        self.smoothed_loss = initial_loss
        self.initialized = False

    def update(self, loss):
        if not self.initialized:
            self.smoothed_loss = loss
            self.initialized = True
        else:
            self.smoothed_loss = self.beta * self.smoothed_loss + (1 - self.beta) * loss

    def get_smoothed_loss(self):
        return self.smoothed_loss


class LossHistory:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def update(self, loss: float):
        self.buffer.append(loss)

    def get_std_dev(self) -> float:
        if len(self.buffer) < 2:  # Need at least 2 data points to calculate std dev
            return float('inf')  # Return a large number or another default value
        return jnp.std(jnp.array(self.buffer))

    def get_recent_losses(self) -> np.array:
        return jnp.array(self.buffer)


def compute_second_order_derivative(gradient_history, step_size=1):
    """
    Approximate second-order derivative using finite differences.

    Parameters:
    gradient_history (list or deque): A history of past gradient values.
    step_size (int): The step size used for finite difference calculation.

    Returns:
    float: Approximated second-order derivative.
    """
    if len(gradient_history) < 2:
        return 0.0  # or some default value, as there's not enough history to compute second derivative

    # Using simple two-point formula for second derivative approximation
    return (gradient_history[-1] - gradient_history[-2]) / step_size


def dynamic_factor(gradient_of_smoothed_loss, step_num, coefficient, tau):
    return 1 / (1 + jnp.abs(gradient_of_smoothed_loss)**2) * (1 + coefficient * (1 - jnp.exp(-step_num / tau)))


def second_order_dynamic_factor(
        gradient_of_smoothed_loss,
        second_order_derivative,
        step_num,
        coefficient,
        tau
        ):
    curvature_factor = 1 / (1 + jnp.abs(second_order_derivative))
    gradient_factor = 1 / (1 + jnp.abs(gradient_of_smoothed_loss)**2)
    time_factor = (1 + coefficient * (1 - jnp.exp(-step_num / tau)))

    return curvature_factor * gradient_factor * time_factor


class ReduceLROnPlateauState(NamedTuple):
    """State for the ReduceLROnPlateau callback."""
    reduce_factor: float
    patience: int
    min_improvement: float
    best_loss: float
    plateau_count: int
    lr: float
    cooldown_counter: int
    cooldown: int


def reduce_on_plateau(
    reduce_factor: float,
    patience: int,
    min_improvement: float,
    cooldown: int,
    lr: float,
): #  -> GradientTransformationWithExtraArgs
    """ Args:
    reduce_factor: Factor by which the learning rate will be reduced.
        new_lr = lr * factor.
    patience: Number of epochs with no improvement after which learning
        rate will be reduced.
    min_improvement: Threshold for measuring the new optimum, to only focus on
        significant changes.
    cooldown: Number of epochs to wait before resuming normal operation
        after lr has been reduced.
    """
    def init_fn(params):
        del params
        return ReduceLROnPlateauState(patience=patience,
                                        reduce_factor=reduce_factor,
                                        min_improvement=min_improvement,
                                        cooldown=cooldown,
                                        cooldown_counter=0,
                                        plateau_count=0,
                                        best_loss=float("inf"),
                                        lr=lr,
                                        )

    def update_fn(
        updates,
        state,
        min_lr=1e-6,
        params=None,
        extra_args={},
    ):
        del params
        current_loss = extra_args.get("loss")

        # Check if the current loss is the best so fa
        best_loss = state.best_loss
        # Update plateau count and check if plateaued
        has_improved = jnp.where(
            (current_loss / best_loss - 1) < -state.min_improvement, 1, 0
        )
        new_best_loss = jnp.where(has_improved, current_loss, best_loss)
        curr_plateau_count = jnp.where(has_improved, 0, state.plateau_count + 1)

        # We're in cooldown, so reduce the counter and ignore any bad epochs
        def in_cooldown():
            new_plateau_count = jnp.array(0, dtype=jnp.int32)  # convert 0 to a JAX array
            new_lr = jnp.array(state.lr, dtype=jnp.float32)  # convert state.lr to a JAX array if it's not already
            new_cooldown_counter = jnp.array(state.cooldown_counter - 1, dtype=jnp.int32)  # convert result to a JAX array
            return new_plateau_count, new_lr, new_cooldown_counter

        # We're not in cooldown, so update the plateau count and lr as usual
        def not_in_cooldown():
            new_plateau_count = jnp.where(
                curr_plateau_count == state.patience, 0, curr_plateau_count
            )
            new_lr = jnp.where(
                curr_plateau_count == state.patience,
                state.lr * state.reduce_factor,
                state.lr,
            )
            new_cooldown_counter = jnp.where(
                curr_plateau_count == state.patience, state.cooldown, 0
            )
            return new_plateau_count, new_lr, new_cooldown_counter

        new_plateau_count, new_lr, new_cooldown_counter = jax.lax.cond(state.cooldown_counter > 0, in_cooldown, not_in_cooldown)
        new_lr = jnp.maximum(new_lr, min_lr)
        updates = jax.tree_util.tree_map(lambda g: new_lr * g, updates)

        new_state = ReduceLROnPlateauState(
            patience=state.patience,
            reduce_factor=state.reduce_factor,
            min_improvement=state.min_improvement,
            plateau_count=new_plateau_count,
            best_loss=new_best_loss,
            lr=new_lr,
            cooldown_counter=new_cooldown_counter,
            cooldown=state.cooldown,
        )
        return updates, new_state

    return init_fn, update_fn

def adjust_learning_rate(
        current_loss,
        previous_loss,
        current_lr,
        momentum_term,
        gradient_of_smoothed_loss,
        step_num,
        coefficient,
        tau,
        exploitation_threshold,
        exploration_threshold,
        second_order_derivative,
        momentum=0.9,
        ):
    # Dynamically calculate factors based on the gradient of smoothed loss
    dynamic_exploitation_factor = 0.5 * dynamic_factor(gradient_of_smoothed_loss, step_num, coefficient, tau)
    # dynamic_exploitation_factor = 0.5 * second_order_dynamic_factor(gradient_of_smoothed_loss, second_order_derivative, step_num, coefficient, tau)
    dynamic_exploration_factor = 1.1 * dynamic_factor(gradient_of_smoothed_loss, step_num, coefficient, tau)

    # Defining the functions for each branch of the conditional
    def exploitation_branch(operand):
        adjust = operand * dynamic_exploitation_factor
        return adjust

    def exploration_branch(operand):
        adjust = operand * dynamic_exploration_factor
        return adjust

    def default_branch(operand):
        adjust = operand * 0.99  # Example gradual reduction factor
        return adjust

    def false_fun_branch(operand):
        # Nested condition inside the false branch
        return jax.lax.cond(jnp.greater(current_loss, previous_loss * 0.99),
        # return jax.lax.cond(jnp.greater(current_loss, previous_loss * exploration_threshold),
                        exploration_branch,
                        default_branch,
                        operand)

    # Creating the conditional
    pred = jnp.less(current_loss, previous_loss * 0.95)  # Example significant drop threshold
    # pred = jnp.less(current_loss, previous_loss * exploitation_threshold)
    adjustment_factor = jax.lax.cond(pred, exploitation_branch, false_fun_branch, current_lr)

    # Update the momentum term
    new_momentum_term = momentum * momentum_term + (1 - momentum) * adjustment_factor
    new_lr = current_lr * new_momentum_term

    return new_lr.astype(float), new_momentum_term


@jax.jit
def reflect_params(new_params, lower_bound, upper_bound):
    def reflect_upper(params):
        diff = upper_bound - params
        return upper_bound + diff

    def reflect_lower(params):
        diff = lower_bound - params
        return lower_bound + diff

    reflected_params = jax.lax.cond(
        jnp.all(new_params > upper_bound),
        reflect_upper,
        lambda x: jax.lax.cond(
            jnp.all(x < lower_bound),
            reflect_lower,
            lambda y: y,  # Identity function, returns new_params unmodified
            x,
        ),
        new_params
    )

    return reflected_params


def wrap_params(new_params, lower_bound, upper_bound):
    range_width = upper_bound - lower_bound
    wrapped_params = lower_bound + (new_params - lower_bound) % range_width
    return wrapped_params
