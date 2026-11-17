from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# Represent the time derivative...
type Matrixdt = npt.NDArray[np.floating]  # as a matrix (for linear systems)
type Vectordt = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]  # As a function


type IntegrationMethod = Literal["forward_euler",]


@runtime_checkable
class StateUpdater(Protocol):
    """Any callable which can update a state vector given a time interval"""
    def __call__(self, x: npt.NDArray[np.floating], dt: float) -> npt.NDArray[np.floating]: ...


class BaseUpdater(ABC):
    """Base class for state updaters.
    Child classes of this implement various methods for
    updating a state vectors x -> x + delta x"""

    def __init__(self, f: Vectordt) -> None:
        self.f = f

    @abstractmethod
    def __call__(self, x: npt.NDArray[np.floating], dt: float) -> npt.NDArray[np.floating]:
        raise NotImplementedError


class ForwardEulerUpdater(BaseUpdater):
    """The simple 'forward Euler' method for simulation.
    Uses the time derivative at each step and updates like
    x -> x + dt * f(x)*x"""

    def __init__(self, f: Vectordt) -> None:
        self.f = f

    def __call__(self, x: npt.NDArray[np.floating], dt: float) -> npt.NDArray[np.floating]:
        step = self.f(x).flatten() if callable(self.f) else (self.f @ x)
        res = x + dt*step
        return res


def make_updater(f: Vectordt, method: IntegrationMethod) -> StateUpdater:
    """Makes a state updater given a time derivative function f(x),
        and a string representing the method to be used"""
    
    match method:
        case "forward_euler":
            return ForwardEulerUpdater(f)
        case _:
            raise ValueError(f"Invalid integration method: {method}")


def make_linear_updater(A: Matrixdt, method: IntegrationMethod) -> StateUpdater:
    """Creates a state updater for a linear system, given the matrix A"""

    return make_updater(
        lambda x: (A @ x),
        method=method
    )
