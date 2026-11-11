from . import formatting, hacks
from .hacks import plot_2D_vector, plot_function
from .settings import setup_matplotlib
from .trajectory import plot_trajectory

__all__ = [
    "setup_matplotlib",
    "hacks",
    "formatting",
    "plot_2D_vector",
    "plot_function",
    "plot_trajectory",
]
