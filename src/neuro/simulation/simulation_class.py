import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .integration import (
    IntegrationMethod,
    StateUpdater,
    make_updater,
)

DEFAULT_TIME_STEP = 1e-6
DEFAULT_METHOD: IntegrationMethod = "forward_euler"


class Simulation:
    def __init__(
            self,
            A: npt.NDArray[np.floating],
            x0: npt.NDArray[np.floating],
            duration: float,
            dt: float=DEFAULT_TIME_STEP,
            method: IntegrationMethod|StateUpdater=DEFAULT_METHOD
        ) -> None:
        """Make a new simulation.
        A: np.ndarray: The matrix A governing the system's dynamics
        x0: The initial state vector
        duration: The total duration of the simulation
        dt: The time step used to calculate each state update
        method: The method used for updating the state.
            Can be either a callable which accepts the state and time step,
            or a string representing a valid IntegrationMethod"""

        self.dimension = len(x0)
        # Convert update matrix and initial state to float and copy them
        self.x0 = x0.astype(float)
        self.A = A.astype(float)
        self.dt = dt
        self.duration = duration
        
        if self.A.shape != (self.dimension, self.dimension):
            raise ValueError(f"System has dimensionality {self.dimension} for A has shape {A.shape}")

        self.update = method if isinstance(method, StateUpdater) else make_updater(self.A, method)
        

    def run(self) -> npt.NDArray[np.floating]:
        """Run the simulation"""
        # Pre-allocate array for storing state history
        n_steps = math.ceil(self.duration / self.dt)
        trajectory = np.full((n_steps + 1, self.dimension), np.nan)

        # Store the initial state
        x = self.x0
        trajectory[0, :] = x
        
        update = self.update
        dt = self.dt

        # Keep updating until the required time has elapsed
        for i in range(1, len(trajectory)):
            x = update(x=x, dt=dt)
            trajectory[i, :] = x

        # Make sure there are now no NaNs in the state history
        if np.any(np.isnan(trajectory)):
            raise RuntimeError("State history contained NaNs - should not happen!")

        return trajectory


def _make_single_callable[**P, S: Simulation](factory: Callable[P, S]) -> Callable[P, npt.NDArray[np.floating]]:
    """Decorator for turning a Simulation class (or factory method for a Simulation instance)
    into a single function which can be called with the same signature to obtain the simulation results"""
    
    def simulate(*args: P.args, **kwargs: P.kwargs) -> npt.NDArray[np.floating]:
        sim = factory(*args, **kwargs)
        res = sim.run()
        return res

    return simulate


simulate = _make_single_callable(Simulation)
