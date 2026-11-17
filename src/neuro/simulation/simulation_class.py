import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, NotRequired, TypedDict, Unpack

import numpy as np
import numpy.typing as npt

from .integration import (
    IntegrationMethod,
    Matrixdt,
    StateUpdater,
    Vectordt,
    make_linear_updater,
    make_updater,
)


class SimKwargs(TypedDict):
    """Keywords which is reused in many places for controlling
    simulations"""

    x0: npt.NDArray[np.floating]
    duration: float
    dt: NotRequired[float]
    method: NotRequired[IntegrationMethod]


class SimulationBase(ABC):
    """Base class for simulations"""

    DEFAULT_METHOD: ClassVar[IntegrationMethod] = "forward_euler"
    DEFAULT_DT: ClassVar[float] = 0.001

    def __init__(self, **kwargs: Unpack[SimKwargs]) -> None:
        self.x0 = kwargs["x0"]
        self.dt = kwargs.get("dt", self.DEFAULT_DT)
        self.duration = kwargs["duration"]
        self.method = kwargs.get("method", self.DEFAULT_METHOD)
        self.update: StateUpdater = self.resolve_method(self.method)

    @abstractmethod
    def resolve_method(self, method: IntegrationMethod) -> StateUpdater:
        raise NotImplementedError

    def run(self) -> npt.NDArray[np.floating]:
        """Run the simulation"""
        dt = self.dt
        updater = self.update
        n_steps = math.ceil(self.duration / dt)
        
        trajectory = np.full((n_steps + 1, len(self.x0)), np.nan)

        # Store the initial state
        x = self.x0
        trajectory[0, :] = x

        # Keep updating until the required time has elapsed
        for i in range(1, len(trajectory)):
            x = updater(x=x, dt=dt)
            trajectory[i, :] = x

        # Make sure there are now no NaNs in the state history
        if np.any(np.isnan(trajectory)):
            raise RuntimeError("State history contained NaNs - should not happen!")

        return trajectory


class LinearSimulation(SimulationBase):
    """A linear simulation, simulating states based on the time derivative
    matrix A"""

    def __init__(self, A: Matrixdt, **kwargs: Unpack[SimKwargs]) -> None:
        self.A = A
        super().__init__(**kwargs)

    def resolve_method(self, method: IntegrationMethod) -> StateUpdater:
        return make_linear_updater(
            A=self.A,
            method=method
        )


class Simulation(SimulationBase):
    """A more general type of simulation, using the time derivative as function
    of x to update states"""

    def __init__(self, f: Vectordt, **kwargs: Unpack[SimKwargs]) -> None:
        self.f = f
        super().__init__(**kwargs)

    def resolve_method(self, method: IntegrationMethod) -> StateUpdater:
        return make_updater(self.f, method)


def _make_single_callable[**P, S: SimulationBase](factory: Callable[P, S]) -> Callable[P, npt.NDArray[np.floating]]:
    """Decorator for turning a Simulation class (or factory method for a Simulation instance)
    into a single function which can be called with the same signature to obtain the simulation results"""
    
    def simulate(*args: P.args, **kwargs: P.kwargs) -> npt.NDArray[np.floating]:
        sim = factory(*args, **kwargs)
        res = sim.run()
        return res

    return simulate


simulate = _make_single_callable(Simulation)
simulate_linear = _make_single_callable(LinearSimulation)
