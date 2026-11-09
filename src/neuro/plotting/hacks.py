"""Stuff for quickly plotting stuff without too much hassle"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes


class Sentinel:
    pass


class NONE(Sentinel):
    pass


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
