import unittest

import numpy as np
import sympy as sp  # type: ignore

from neuro.dynamical_system import DynamicalSystem
from neuro.simulation import simulate, simulate_linear


class TestRomeoJuliet(unittest.TestCase):
    a = 1.0
    b = 1.0
    J0 = 1.0
    R0 = 1.0
    duration = 5.0

    def setUp(self) -> None:
        self.A = np.array([[0, self.a], [-self.b, 0]], dtype=float)

        R, J = sp.symbols("R, J")
        self.system = DynamicalSystem(
            equations=[
                self.a*J,
                -self.b*R
            ],
            symbols=(R, J),
        )

        self.x0 = np.array([self.R0, self.J0])
        return super().setUp()

    def test_linear_simulation_stable(self) -> None:
        """Runs the 'lovers' simulation where they are supposed to orbit the origin.
        Check that the distance to the origin remains fairly constant"""

        res = simulate_linear(
            self.A,
            x0=self.x0,
            duration=self.duration
        )
        initial_dist = sum(v**2 for v in self.x0)**.5
        dists = np.linalg.norm(res, axis=1)
        self.assertTrue(np.allclose(dists, initial_dist, atol=1e-2))

    def test_general_simulation_matches_linear(self) -> None:
        lin = simulate_linear(
            self.A,
            x0=self.x0,
            duration=self.duration
        )

        gen = simulate(
            self.system.f_vec,
            x0=self.x0,
            duration=self.duration
        )

        # Check that the shapes match
        self.assertTupleEqual(lin.shape, gen.shape)
        
        # Check that the simulated trajectories match
        self.assertTrue(np.allclose(lin, gen))
    