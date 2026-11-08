from importlib.metadata import version

__version__ = version("neuro")


from neuro.plotting import setup_matplotlib

__all__ = [
    'setup_matplotlib'
]
