from . import formatting, hacks, phase_diagrams, pyplot_utils
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
    "phase_diagrams",
    "pyplot_utils"
]
