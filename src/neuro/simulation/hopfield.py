from collections import deque
from collections.abc import Iterable, Iterator
from functools import partial
from typing import ClassVar

import numpy as np
import numpy.typing as npt

type Randomizer = np.random.RandomState|int|None


def _as_randomstate(rs: Randomizer) -> np.random.RandomState:
    """Helper for converting one of various types into a randomstate object."""
    if not isinstance(rs, np.random.RandomState):
        rs = np.random.RandomState(rs)
    return rs


def scramble_vector(
        vector: npt.NDArray[np.floating],
        noise: float,
        randomstate: Randomizer=None
        ) -> npt.NDArray[np.floating]:
    rs = _as_randomstate(randomstate)
    noise_: npt.NDArray[np.floating] = rs.choice(
        [+1, -1],
        p=[1.0 - noise, noise],
        size=vector.shape,
        replace=True
    )

    return vector * noise_


class HopfieldNetwork:
    # Values that encode neurons firing/not firing
    SIGNALS: ClassVar[tuple[int, ...]] = (+1, -1)

    def __init__(
            self,
            concepts: npt.NDArray[np.floating],
            randomstate: Randomizer=None
            ) -> None:
        """Initializes a Hopfield network, using the input concepts.
        randomstate can be a numpy RandomState, or seed (int/None) used for random data"""

        self.rs = _as_randomstate(randomstate)

        self.concepts = concepts.copy()
        self.n_concepts, self.dim = self.concepts.shape
        self.W = weights_from_concepts(self.concepts)
        self.update_single = partial(hopfield_update_step, self.W)

    def energy(self, state: npt.NDArray[np.floating]) -> float:
        res = compute_hopfield_energy(self.W, state)
        return res

    def scramble_concepts(self, noise_ratio: float) -> npt.NDArray[np.floating]:
        """Applies some noise to the concept vectors"""

        return scramble_vector(self.concepts, noise=noise_ratio, randomstate=self.rs)

    @classmethod
    def _get_random(cls, dim: int, rs: np.random.RandomState, n: int|None=None) -> npt.NDArray[np.floating]:
        size = dim if n is None else (n, dim)
        res = rs.choice(cls.SIGNALS, replace=True, size=size)
        return res
    
    def get_random_state(self, n: int|None=None) -> npt.NDArray[np.floating]:
        return self._get_random(dim=self.dim, rs=self.rs, n=n)

    def _iter_gradient_descent(
            self,
            state: npt.NDArray[np.floating],
            n_steps_max: int|None=None,
        ) -> Iterator[npt.NDArray[np.floating]]:
        """Iteratively performs update steps to minimize the Hopfield energy.
        Each step yields the state at the current update step"""

        state = state.copy()
        E = self.energy(state)
        nits = 0
        stopat = float("inf") if n_steps_max is None else n_steps_max
        while nits <= stopat:
            yield state
            state = hopfield_update_step(W=self.W, state=state)
            E_new = self.energy(state)
            delta = E_new - E
            if delta >= 0.0:
                break
            E = E_new
            nits += 1

    def update_repeated(
            self,
            state: npt.NDArray[np.floating],
            n_steps_max: int|None=None,
            return_full_history: bool=False
            ) -> npt.NDArray[np.floating]:
        """Performs update steps until n_steps_max is reached, or the energy stabilizes"""

        state_iter = self._iter_gradient_descent(state=state, n_steps_max=n_steps_max)

        if return_full_history:
            return np.stack(list(state_iter), axis=0)
        else:
            return deque(state_iter, maxlen=1).pop()

    @classmethod
    def init_random(cls, dim: int, n_concepts: int, randomstate: Randomizer=None) -> HopfieldNetwork:
        rs = _as_randomstate(randomstate)
        concepts = cls._get_random(dim=dim, n=n_concepts, rs=rs)
        inst = HopfieldNetwork(concepts, randomstate=rs)
        return inst

    def __and__(self, other: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        if not isinstance(other, np.ndarray):
            return NotImplemented
        
        return compute_overlap(self.concepts, other)


def compute_overlap(bases: npt.NDArray[np.floating], states: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Computes the overlap (averaged pairwise product) of vectors.
    Takes shapes n x d and m x d and returns shape n x m"""
    normfac: float = 1/states.shape[-1]
    return normfac*(bases @ states.T)


def weights_from_concepts(*xis: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Computes the Hopfield weight matrix from input concepts w_ij = <xi_i*xi_j> (averaged over concepts)
    Input can be either a (m, n) array, or m separate (n,) arrays, where m is the number of
    concept vectors xi, and n is the dimensionality/number of neurons."""

    concepts = xis[0] if len(xis) == 1 else np.vstack(xis)
    _, n_neurons = concepts.shape

    norm_fac: float = 1.0/n_neurons
    res = norm_fac*np.sum([np.outer(xi, xi) for xi in concepts], axis=0)
    np.fill_diagonal(res, 0)
    return res


def compute_hopfield_energy(W: npt.NDArray[np.floating], s: Iterable[float]|npt.NDArray[np.floating]) -> float:
    """Computes the Hopfield energy -0.5 sum w_ij s_i s_j"""
    if not isinstance(s, np.ndarray):
        s = np.fromiter(s, dtype=float)

    prod_ = -0.5*np.sum(s @ W * s, axis=-1)
    res = np.sum(prod_) if isinstance(prod_, np.ndarray) else prod_
    
    return res.item()


def hopfield_update_step(
        W: npt.NDArray[np.floating],
        state: npt.NDArray[np.floating]
        ) -> npt.NDArray[np.floating]:
    """Performs a single update step s -> sign(W*s), keeping values where sign = 0"""

    state = state.copy()

    res = np.sign(state @ W)
    # Keep the original values where the update rule gives zero
    res[res == 0] = state[res == 0]
    return res



def simulate_retrieval(
    p: int,
    n: int=200,
    noise: float=0.10,
    seed: int|None=None,
    n_trials: int=1
    ) -> npt.NDArray[np.floating]:
    """Constructs a Hopfield network with n neurons and p concepts.
    Applies noise to the first concept, attempts retrieval, and returns the overlap
    between the retrieved state and the original concept."""

    overlaps: list[float] = []
    rs = _as_randomstate(seed)
    for _ in range(n_trials):
        hopfnet = HopfieldNetwork.init_random(dim=n, n_concepts=p, randomstate=rs)
        target = hopfnet.concepts[0]

        scrambled = scramble_vector(
            target,
            randomstate=hopfnet.rs,
            noise=noise
        )

        recovered = hopfnet.update_repeated(scrambled)
        overlap = compute_overlap(target, recovered).item()
        overlaps.append(overlap)

    return np.array(overlaps)
