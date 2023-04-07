import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

import haiku as hk
import numpy as np
import tensorflow_datasets as tfds

from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)
Array = jnp.ndarray
Batch = Mapping[str, np.ndarray]

def jax_lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    if jnp.isscalar(A):
        A = A * jnp.ones(dimensions)
        return A
    shape = tuple(dimensions) + A.shape
    A = A[jnp.newaxis, ...]
    A = jnp.broadcast_to(A, shape)
    return A


def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
    """Helper function for loading and preparing tfds splits."""
    ds = split
    ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1000)
    ds = ds.repeat()
    return iter(tfds.as_numpy(ds))


def prepare_tf_dataset(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
    """Helper function for preparing tfds splits for use in fliax."""
    # TODO: add length arguments to function.
    # Batch is [y, thetas, d]
    data = batch.astype(np.float32)
    x = data[:, :len_x]
    cond_data = data[:, len_x:]
    theta = cond_data[:, :-len_x]
    d = cond_data[:, -len_x:-len_xi]
    xi = cond_data[:, -len_xi:]
    return x, theta, d, xi


@hk.without_apply_rng
@hk.transform
def log_prob(data: Array, theta: Array, xi: Array) -> Array:
    # Get batch
    shift = data.mean(axis=0)
    scale = data.std(axis=0) + 1e-14

    model = make_nsf(
        event_shape=EVENT_SHAPE,
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins,
        standardize_x=True,
        standardize_theta=False,
        use_resnet=True,
        event_dim=EVENT_DIM,
        shift=shift,
        scale=scale,
    )
    return model.log_prob(data, theta, xi)


# TODO: Finish this helper function by making dependent functions.
def pairplot(
    samples: Union[List[jnp.ndarray], jnp.ndarray],
    points: Optional[
        Union[List[jnp.ndarray], jnp.ndarray]
    ] = None,
    limits: Optional[Union[List, jnp.ndarray]] = None,
    subset: Optional[List[int]] = None,
    upper: Optional[str] = "hist",
    diag: Optional[str] = "hist",
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Union[List, jnp.ndarray] = [],
    points_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"],
    fig=None,
    axes=None,
    **kwargs,
):
    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)

    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    samples, dim, limits = prepare_for_plot(samples, limits)

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    diag_func = get_diag_func(samples, limits, opts, **kwargs)

    def upper_func(row, col, limits, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["upper"][n] == "hist" or opts["upper"][n] == "hist2d":
                    hist, xedges, yedges = np.histogram2d(
                        v[:, col],
                        v[:, row],
                        range=[
                            [limits[col][0], limits[col][1]],
                            [limits[row][0], limits[row][1]],
                        ],
                        **opts["hist_offdiag"],
                    )
                    plt.imshow(
                        hist.T,
                        origin="lower",
                        extent=(
                            xedges[0],
                            xedges[-1],
                            yedges[0],
                            yedges[-1],
                        ),
                        aspect="auto",
                    )

                elif opts["upper"][n] in [
                    "kde",
                    "kde2d",
                    "contour",
                    "contourf",
                ]:
                    density = gaussian_kde(
                        v[:, [col, row]].T,
                        bw_method=opts["kde_offdiag"]["bw_method"],
                    )
                    X, Y = np.meshgrid(
                        np.linspace(
                            limits[col][0],
                            limits[col][1],
                            opts["kde_offdiag"]["bins"],
                        ),
                        np.linspace(
                            limits[row][0],
                            limits[row][1],
                            opts["kde_offdiag"]["bins"],
                        ),
                    )
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(density(positions).T, X.shape)

                    if opts["upper"][n] == "kde" or opts["upper"][n] == "kde2d":
                        plt.imshow(
                            Z,
                            extent=(
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ),
                            origin="lower",
                            aspect="auto",
                        )
                    elif opts["upper"][n] == "contour":
                        if opts["contour_offdiag"]["percentile"]:
                            Z = probs2contours(Z, opts["contour_offdiag"]["levels"])
                        else:
                            Z = (Z - Z.min()) / (Z.max() - Z.min())
                        plt.contour(
                            X,
                            Y,
                            Z,
                            origin="lower",
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            colors=opts["samples_colors"][n],
                            levels=opts["contour_offdiag"]["levels"],
                        )
                    else:
                        pass
                elif opts["upper"][n] == "scatter":
                    plt.scatter(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["scatter_offdiag"],
                    )
                elif opts["upper"][n] == "plot":
                    plt.plot(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["plot_offdiag"],
                    )
                else:
                    pass

    return _arrange_plots(
        diag_func, upper_func, dim, limits, points, opts, fig=fig, axes=axes
    )
