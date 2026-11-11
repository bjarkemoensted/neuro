from importlib.metadata import version

__version__ = version("neuro")

from . import plotting, sentinels
from .plotting import setup_matplotlib
from .simulation import simulate

__all__ = [
    'setup_matplotlib',
    "simulate",
    "plotting",
    "sentinels",
]
