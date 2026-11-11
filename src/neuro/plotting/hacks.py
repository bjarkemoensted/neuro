"""Stuff for quickly plotting stuff without too much hassle"""

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from neuro.sentinels import NONE

# The allowed types for matplotlib colors :'(
type ColorType = (
    tuple[float, float, float]
    | str
    | tuple[float, float, float, float]
    | tuple[tuple[float, float, float] | str, float]
    | tuple[tuple[float, float, float, float], float]
)


type functype = Callable[[float], float] | Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]


def _infer_function_name(func: functype) -> str:
    res = getattr(func, '__name__', func.__class__.__name__)
    return res


@dataclass
class FunctionProperties:
    name: str
    free_var_name: str

    @classmethod
    def infer(cls, func: functype) -> Self:
        sig = inspect.signature(func)
        varname = next(iter(sig.parameters))
        funcname = _infer_function_name(func)
        inst = cls(name=funcname, free_var_name=varname)
        return inst


def plot_function(
        func: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
        ax: Axes|None=None,
        n_points: int=10_000,
        xmin: float = 0.0,
        xmax: float = 10,
        xlabel: str|None|type[NONE]=NONE,
        ylabel: str|None|type[NONE]=None,
        label: str|None=None
        ) -> Axes:

    if ax is None:
        _, ax = plt.subplots()

    f = func if isinstance(func, np.ufunc) else np.vectorize(func)
    x = np.linspace(xmin, xmax, num=n_points)
    y = f(x)

    inferred = FunctionProperties.infer(func)

    if xlabel is not None and ax.get_xlabel() == "":
        ax.set_xlabel(xlabel if isinstance(xlabel, str) else inferred.free_var_name)

    if ylabel is not None and ax.get_ylabel() == "":
        ax.set_ylabel(ylabel if isinstance(ylabel, str) else inferred.name)

    ax.plot(x, y, label=label)

    return ax


def _grab_color(ax: Axes) -> ColorType:
    """Hack for grabbing the next color in the color cycle for an axis, and incrementing
    the cycle.
    There's no public-facing way of doing this, so we plot a non-existing line segment, grab its
    color, and remove it again."""
    line, = ax.plot([], [])
    color = line.get_color()
    line.remove()
    return color


def plot_2D_vector(
    vec: npt.NDArray[np.floating|np.int_]|Sequence[float],
    ax: Axes|None=None,
    label: str|None=None,
    color: ColorType|None|type[NONE]=NONE,
    quiver_kwargs: dict[str, Any]|None=None
    ) -> Axes:
    
    v = vec if isinstance(vec, np.ndarray) else np.array(vec)
    if v.shape != (2,):
        raise ValueError("Input vector must be 2-dimensional")

    if ax is None:
        _, ax = plt.subplots()

    # If no color is specified, use the next in the ax' cycle
    if isinstance(color, type) and issubclass(color, NONE):
        color = _grab_color(ax)

    _kwargs = {} if quiver_kwargs is None else quiver_kwargs
    ax.quiver(
        0,
        0,
        v[0],
        v[1],
        angles='xy',
        scale_units='xy',
        scale=1,
        label=label,
        color=color,
        **_kwargs
    )

    low, high = min(v), max(v)
    span = high - low
    margin = 0.02
    epsilon = margin*span
    low -= epsilon
    high += epsilon

    for getter, setter in ((ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim)):
        a, b = getter()

        a_prime = min(a, low)
        b_prime = max(b, high)
        setter(a_prime, b_prime)

    ax.set_aspect('equal')

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Put ticks on the axes
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return ax
