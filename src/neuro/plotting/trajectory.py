import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

DEFAULT_COLORMAP = "Grays"


def plot_trajectory(
        arr: npt.NDArray[np.floating],
        ax: Axes|None=None,
        cmap: str|Colormap|None=DEFAULT_COLORMAP,
        colorbar: bool=True,
    ) -> Axes:
    """Plots a history of state vectors.
    arr: (N, 2) array with the state history"""

    # TODO add option to plot arrows connecting every N'th point or everytime a fixed distance has elapsed
    n_points, dim = arr.shape
    if dim != 2:
        raise ValueError(f"Need 2-dimensional data - got shape {arr.shape}")
    
    if ax is None:
        _, ax = plt.subplots()

    x, y = arr.T
    sc = ax.scatter(
        x=x,
        y=y,
        c=np.arange(n_points),
        cmap=cmap,
        marker=".",
        edgecolors='none'
    )

    if colorbar:
        cbar = ax.figure.colorbar(sc, ax=ax, label='N time steps')
        cbar.outline.set_visible(False)

    return ax
