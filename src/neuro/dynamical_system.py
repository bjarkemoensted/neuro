from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Unpack

import numpy as np
import numpy.typing as npt
import sympy as sp  # type: ignore

from neuro.simulation.simulation_class import SimKwargs, simulate, simulate_linear
from neuro.types import FixedPointCategory

type Point = dict[sp.Symbol, float]


@dataclass
class FixedPoint:
    coords: tuple[float, ...]
    category: FixedPointCategory
    stable: bool


def determine_stability(J: sp.Matrix, point: Point) -> tuple[bool, FixedPointCategory]:
    """Given a Jacobian matrix, and a fixed point at which to evaluate it, determines the stability
    of the point"""

    local = J.subs(point)
    
    counts = {k: 0 for k in (-1, 0, +1)}
    zero = sp.Integer(0)  # sympy Symbols freak out if we compare them to ordinary peasant integers!

    # eigenvals gives a dict mapping values to multiplicities, so just take the keys
    for eigenval in local.eigenvals():
        realpart = sp.re(eigenval)
        if realpart == zero:
            counts[0] += 1
        elif realpart < zero:
            counts[-1] += 1
        else:
            assert realpart > zero
            counts[+1] += 1

    if counts[0] > 0:
        raise NotImplementedError("Encountered eigenvalue with zero real part")
    elif counts[+1] == 0:
        return True, "stable"
    elif counts[-1] == 0:
        return False, "unstable"
    elif counts[+1] > 0 and counts[-1] > 0:
        return False, "saddle"
    else:
        raise RuntimeError(f"Invalid eigenvalue sign counts: {counts}")
    

def _extract_variables(*equations: sp.Expr) -> set[sp.Symbol]:
    """Returns the set of free symbols that occur in any of the input equations"""
    symbols = (symbol for eqn in equations for symbol in eqn.free_symbols)
    res: set[sp.Symbol] = set()
    for symbol in symbols:
        assert isinstance(symbol, sp.Symbol)
        res.add(symbol)
    
    return res


def determine_nullclines(*equations: sp.Expr) -> tuple[tuple[dict[sp.Symbol, sp.Expr]], ...]:
    res = tuple(
        tuple(sp.solve(eqn))
        for eqn in equations
    )

    return res


def determine_fixed_points(*equations: sp.Expr) -> tuple[dict[sp.Symbol, float], ...]:
    solutions = sp.solve(equations)
    return tuple(solutions)


class DynamicalSystem:
    def __init__(
            self,
            equations: Iterable[sp.Expr|float],
            symbols: Sequence[sp.Symbol]|None=None
            ) -> None:

        self.equations = [eq if isinstance(eq, sp.Expr) else sp.sympify(eq) for eq in equations]
        self.f = sp.Matrix(self.equations)

        symbols_iter = sorted(_extract_variables(*self.equations), key=str) if symbols is None else symbols
        self.symbols: tuple[sp.Symbol, ...] = tuple(symbols_iter)
        self.symbols_vec = sp.Matrix([sym for sym in self.symbols])
        self._f_numpy = sp.lambdify([self.symbols_vec], self.f, modules="numpy")
        # time derivative, but taking separate arguments for each variable, useful for mesh grids
        self.f_unpacked = sp.lambdify(self.symbols, self.equations, modules="numpy")

    def f_vec(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Returns the local time derivative as a flat numpy array, i.e. with shape (N, 1)"""
        res: npt.NDArray[np.floating] = self._f_numpy(x).flatten()
        assert len(res.shape) == 1
        return res

    @cached_property
    def nullclines(self) -> tuple[tuple[dict[sp.Symbol, sp.Expr]], ...]:
        return determine_nullclines(*self.equations)

    @cached_property
    def dim(self) -> int:
        return len(self.equations)

    @cached_property
    def fixed_points(self) -> tuple[Point, ...]:
        return determine_fixed_points(*self.equations)

    @cached_property
    def fixed_points_coords(self) -> list[tuple[float, ...]]:
        res = [tuple(fp[sym] for sym in self.symbols) for fp in self.fixed_points]
        return res

    def point_as_coords(self, point: Point) -> tuple[float, ...]:
        return tuple(point[symbol] for symbol in self.symbols)

    @cached_property
    def jacobian(self) -> sp.Matrix:
        res = self.f.jacobian(self.symbols)
        return res

    def evaluate_jacobian(self, point: Point) -> npt.NDArray[np.floating]:
        A_local = self.jacobian.subs(point)
        res = np.array(A_local).astype(float)
        return res

    def iter_fixed_points(self) -> Iterator[FixedPoint]:
        for p in self.fixed_points:
            stable, category = determine_stability(J=self.jacobian, point=p)
            yield FixedPoint(
                coords = self.point_as_coords(p),
                stable=stable,
                category=category
            )

    def is_linear(self) -> bool:
        for eqn in self.equations:
            for symbol in self.symbols:
                deg = sp.Poly(eqn, symbol).degree()
                if deg > 1:
                    return False
        return True

    def simulate(self, **kwargs: Unpack[SimKwargs]) -> npt.NDArray[np.floating]:
        if self.is_linear():
            return simulate_linear(
                A=np.array(self.f).astype(float),
                **kwargs
            )
        else:
            return simulate(
                f=self.f_vec,
                **kwargs
            )
    