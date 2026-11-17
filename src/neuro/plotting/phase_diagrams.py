import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sympy as sp  # type: ignore
from matplotlib.axes import Axes

from neuro.plotting.formatting import label_with_values
from neuro.plotting.pyplot_utils import cycle_color
from neuro.types import ColorType, FixedPointCategory, is_fixed_point_category

if TYPE_CHECKING:
    from neuro.dynamical_system import DynamicalSystem



def _make_mesh(limits: tuple[tuple[float, float], ...], n_points: int) -> tuple[npt.NDArray[np.floating], ...]:
    """Make a meshgrid with n points along each axis, and with the specified limits,
    format: ((xmin, xmax), ...)"""
    grid = np.meshgrid(
        *list(np.linspace(a, b, n_points, dtype=float) for a, b in limits),
        indexing="ij"
    )
    return grid


def suggest_bounding_box(
        system: DynamicalSystem,
        square: bool=True,
        buffer_percent: float=5.0
        ) -> tuple[tuple[float, float], ...]:
    """Returns a bounding box, e.g. ((x_low, x_high), ...) which contains
    all fixe points, plus a little buffer.
    if square = True, forces the bounds for all coordinates to be identical"""

    # Determine min/max of the coordinate values for a fixed point
    ordered = map(sorted, zip(*system.fixed_points_coords, strict=True))
    limits = [(arr[0], arr[-1]) for arr in ordered]
    limits.sort(key = lambda t: t[1] - t[0])

    # Compute the max distance between fixed points along axes
    spans = [b - a for a, b in limits]
    max_span = spans[-1]
    min_span = next(span for span in spans if span > 0)

    limits = [(a, a + max_span) for a, _ in limits] if square else [(a, max(b, a+min_span)) for a, b in limits]

    # Add some padding
    for i, (a, b) in enumerate(limits):
        eps = (b-a)*(buffer_percent/100)
        limits[i] = (a - eps, b + eps)


    return tuple(limits)


def _resolve_limits(
        limits: tuple[float, float]|tuple[tuple[float, float], ...]|None,
        system: DynamicalSystem,
    ) -> tuple[tuple[float, float], ...]:
    """This is just for allowing 'limits' to be either a single tuple
    of lower and upper bounds, to be used on all coordinates, or a tuple thereof.
    A value of None is resolved by allowing the DynamicalSystem instance to decide"""
    
    if limits is None:
        return suggest_bounding_box(system)

    if not isinstance(limits[0], tuple):
        a, b = limits
        assert not isinstance(a, tuple) and not isinstance(b, tuple)
        return tuple((a, b) for _ in range(system.dim))

    res_list: list[tuple[float, float]] = []
    for t in limits:
        assert isinstance(t, tuple)
        res_list.append(t)

    return tuple(res_list)


def plot_nullclines(
        system: DynamicalSystem,
        ax: Axes,
        symbols: tuple[sp.Symbol, sp.Symbol],
        limits: tuple[tuple[float, float], ...],
        n_points: int=50,
        color: Iterable[ColorType]|None=None
    ) -> None:

    grid = _make_mesh(limits=limits, n_points=n_points)
    X, Y = grid

    color_it = None if color is None else iter(color)

    for nullcline in system.nullclines:
        color_ = cycle_color(ax) if color_it is None else next(color_it)
        for solution in nullcline:
            sym, val = next(iter(solution.items()))
            expr: sp.Expr = sp.Add(sym, -val)
            f = sp.lambdify(symbols, expr, "numpy")
            ax.contour(
                X,
                Y,
                f(X, Y),
                levels=[0],
                colors=color_,
            )
            label = label_with_values(solution)
            plt.plot([], [], color=color_, label=label)


def plot_fixed_points(
        system: DynamicalSystem,
        ax: Axes,
        color: ColorType="black",
    ) -> None:

    fixed_points = sorted(system.iter_fixed_points(), key=lambda fp: fp.category)
    markers: dict[FixedPointCategory, str] = {
        "stable": "o",
        "saddle": "+",
        "unstable": "x",
        "non-hyperbolic": "d",
    }

    for cat, fps in itertools.groupby(fixed_points, key = lambda fp: fp.category):
        assert is_fixed_point_category(cat)
        marker = markers[cat]
        label = f"Fixed point ({cat})"
        ax.scatter(
            *zip(*(fp.coords for fp in fps), strict=True),
            zorder=99,
            marker=marker,
            color=color,
            label=label
        )


def plot_vector_field(
        system: DynamicalSystem,
        ax: Axes,
        limits: tuple[tuple[float, float], ...],
        n_points: int=20,
        alpha: float=0.5,
        ) -> None:
    grid = _make_mesh(limits=limits, n_points=n_points)
    X, Y = grid
    U, V = system.f_unpacked(*grid)

    magnitude = np.hypot(U, V)

    U_norm = U / np.maximum(magnitude, 1e-10)
    V_norm = V / np.maximum(magnitude, 1e-10)

    ax.quiver(
        X,
        Y,
        U_norm,
        V_norm,
        alpha=alpha,
        pivot="middle"
    )


def plot_phase_diagram(
        system: DynamicalSystem,
        ax: Axes|None=None,
        nullclines: bool=True,
        fixed_points: bool=True,
        vector_field: bool=True,
        limits: tuple[float, float]|tuple[tuple[float, float], ...]|None=None,
        xlabel: str|None=None,
        ylabel: str|None=None,
        legend: bool=True,
    ) -> Axes:

    if ax is None:
        _, ax = plt.subplots()
    
    limits = _resolve_limits(limits=limits, system=system)

    s1, s2 = system.symbols
    ax.set_xlabel(str(s1) if xlabel is None else xlabel)
    ax.set_ylabel(str(s2) if ylabel is None else ylabel)

    if nullclines:
        plot_nullclines(
            system=system,
            ax=ax,
            symbols=(s1, s2),
            limits=limits
        )

    if fixed_points:
        plot_fixed_points(
            system=system,
            ax=ax
        )

    if vector_field:
        plot_vector_field(
            system=system,
            ax=ax,
            limits=limits,
        )

    if legend:
        ax.legend()
    return ax
