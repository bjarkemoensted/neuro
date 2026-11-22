import unittest
from typing import ClassVar

import numpy as np

from neuro.simulation import hopfield


class TestHopfield(unittest.TestCase):
    DIM: ClassVar[int] = 50
    N_CONCEPTS: ClassVar[int] = 3

    def setUp(self) -> None:
        self.hopfnet = hopfield.HopfieldNetwork.init_random(
            n_concepts=self.N_CONCEPTS,
            dim=self.DIM,
            randomstate=0
        )
        self.W = self.hopfnet.W
        self.concepts = self.hopfnet.concepts
        self.vecs = [self.hopfnet.get_random_state() for _ in range(10)]
        self.vecs_multi = np.vstack([v.copy() for v in self.vecs])

        return super().setUp()

    def test_compute_hopfield_energy(self) -> None:
        """Checks that computing the Hopfield energies seems to work"""
        e_total = 0.0
        for vec in self.vecs:
            e_hopf = hopfield.compute_hopfield_energy(self.W, vec)
            self.assertIsInstance(e_hopf, float)
            e_total += e_hopf

        e_multi = hopfield.compute_hopfield_energy(self.W, self.vecs_multi)
        self.assertAlmostEqual(e_multi, e_total)

    def test_detuning_from_concept_increases_energy(self) -> None:
        """Checks that the Hopfield energy of its stored concepts are local minima"""
        for xi in self.concepts:
            e_c = hopfield.compute_hopfield_energy(self.W, xi) 
            for i in range(len(xi)):
                vec = xi.copy()
                vec[i] *= -1
                e_perturbed = hopfield.compute_hopfield_energy(self.W, vec)
                self.assertGreaterEqual(e_perturbed, e_c)

    def test_concept_recovery(self) -> None:
        """Check that adding a bit of noise to the Hopfield net's concepts, then applying
        gradient descent, will recover the concepts"""

        scrambled = self.hopfnet.scramble_concepts(noise_ratio=0.10)
        for mu in range(self.hopfnet.n_concepts):
            vec = scrambled[mu]
            target = self.hopfnet.concepts[mu]
            recovered = self.hopfnet.update_repeated(vec)
            self.assertTrue(np.all(target == recovered))

    def test_update(self) -> None:
        """Check state updating works, and is consistent across single/multi state update"""

        vecs = [self.hopfnet.get_random_state() for _ in range(5)]
        vecs_multi = np.vstack([v.copy() for v in vecs])

        update_individual = np.stack([self.hopfnet.update_single(v) for v in vecs])
        update_multi = self.hopfnet.update_single(vecs_multi)
        self.assertTrue(np.all(update_individual == update_multi))
