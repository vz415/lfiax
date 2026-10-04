import jax
import jax.numpy as jnp
from typing import NamedTuple


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
