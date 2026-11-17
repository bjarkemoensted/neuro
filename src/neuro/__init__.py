from importlib.metadata import version

__version__ = version("neuro")

from . import plotting, sentinels, simulation
from .dynamical_system import DynamicalSystem
from .plotting import setup_matplotlib

__all__ = [
    'setup_matplotlib',
    "simulation",
    "plotting",
    "sentinels",
    "DynamicalSystem",
]
