from matplotlib.axes import Axes

from neuro.types import ColorType


def cycle_color(ax: Axes) -> ColorType:
    """Hack for grabbing the next color in the color cycle for an axis, and incrementing
    the cycle.
    There's no public-facing way of doing this, so we plot a non-existing line segment, grab its
    color, and remove it again."""
    line, = ax.plot([], [])
    color = line.get_color()
    line.remove()
    return color
