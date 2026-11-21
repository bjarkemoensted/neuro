import math
import unittest
from dataclasses import dataclass

from neuro.computation import nernst


def nernst_simple(c_in: float, c_out: float, z: int) -> float:
    """Implements the simplified formula for Nernst potential at body temperature from
    eq. (10) the the compendium for week 2.
    Returns the Nernst potential in mV"""

    C = 61.5  # mV - this is just RT/F at body temperate, times a log 10 conversion factor
    res = (C / z)*math.log10(c_out / c_in)
    return res


@dataclass
class Electrolyte:
    ion: str
    c_in: float
    c_out: float
    z: int


electrolyte_examples = (
    Electrolyte("Na+", 15, 145, 1),
    Electrolyte("K+", 140, 5, 1),
    Electrolyte("Cl-", 10, 110, -1),
    Electrolyte("Ca2+", 0.0001, 2, 2),
)


class TestNernst(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = electrolyte_examples
        return super().setUp()
    
    def test_nernst_potential(self) -> None:
        for ex in self.examples:
            print(ex)
            res_simple = nernst_simple(ex.c_in, ex.c_out, ex.z)
            res = nernst.nernst_potential(ex.c_in, ex.c_out, ex.z).to("mV").magnitude
            # Check that they're within 1mV of each other
            self.assertAlmostEqual(res, res_simple, delta=1)
