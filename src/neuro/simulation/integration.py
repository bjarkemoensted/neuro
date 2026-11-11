from functools import partial
from typing import Literal, Protocol, TypeGuard, get_args, runtime_checkable

import numpy as np
import numpy.typing as npt

type IntegrationMethod = Literal["forward_euler",]
_integration_methods = get_args(IntegrationMethod.__value__)


def is_integration_method(s: object) -> TypeGuard[IntegrationMethod]:
    return s in _integration_methods


@runtime_checkable
class StateUpdater(Protocol):
    """Any callable which can update a state vector given a time interval"""
    def __call__(self, x: npt.NDArray[np.floating], dt: float) -> npt.NDArray[np.floating]: ...


def forward_euler_update(
        A: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating],
        dt: float
    ) -> npt.NDArray[np.floating]:
    """Performs a simple forward-Euler update step, i.e.
    x -> dt*A*x"""

    delta = dt*(A @ x)
    res = x + delta
    return res


def make_updater(
        A: npt.NDArray[np.floating],
        method: IntegrationMethod
        ) -> StateUpdater:
    """Take the name of an integration method, and returns a callable which can update
    a state vector"""

    match method:
        case "forward_euler":
            return partial(forward_euler_update, A=A)
        case _:
            raise ValueError(f"Invalid integration method: {method:r!}")
