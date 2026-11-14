import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

DEFAULT_COLORMAP = "Grays"


def plot_trajectory(
    arr: npt.NDArray[np.floating],
    ax: Axes | None = None,
    cmap: str | Colormap | None = DEFAULT_COLORMAP,
    colorbar: bool = True,
    markersize: float = 4.0,
) -> Axes:
    """Plots a history of 2D state vectors."""

    n_points, dim = arr.shape
    if dim != 2:
        raise ValueError(f"Need 2-dimensional data - got shape {arr.shape}")

    if ax is None:
        _, ax = plt.subplots()

    x, y = arr.T

    # Determine sampling interval from plot dimensions and marker size.
    x_range = np.ptp(x)
    y_range = np.ptp(y)

    bbox = ax.get_window_extent()
    width, height = bbox.width, bbox.height

    # Approximate point spacing in display coordinates (pixels).
    dx = x_range * width / max(n_points, 1)
    dy = y_range * height / max(n_points, 1)

    spacing = markersize * plt.rcParams["figure.dpi"] / 72 * 2

    N = max(1, int(np.ceil(spacing / max(min(dx, dy), 1e-12))))

    indices = np.arange(0, n_points, N)

    # Scatter sampled points.
    sc = ax.scatter(
        x[indices],
        y[indices],
        c=indices,
        cmap=cmap,
        marker=".",
        s=markersize**2,
        edgecolors="none",
    )

    # Connect consecutive sampled points with colored arrows.
    norm = sc.norm
    colors = sc.cmap(norm(indices[:-1]))

    for i, color in enumerate(colors):
        j, k = indices[i], indices[i + 1]

        ax.annotate(
            "",
            xy=(x[k], y[k]),
            xytext=(x[j], y[j]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.0,
                shrinkA=2,
                shrinkB=2,
            ),
        )

    if colorbar:
        cbar = ax.figure.colorbar(sc, ax=ax, label="N time steps")
        cbar.outline.set_visible(False)

    return ax