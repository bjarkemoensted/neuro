import math
from typing import Literal

from pint import Quantity

from neuro import constants

CELSIUS_OFFSET = 273.15
BODY_TEMP_C = 37.0
BODY_TEMP_KELVIN = BODY_TEMP_C + CELSIUS_OFFSET


def nernst_potential(
        c_in: float,
        c_out: float,
        z: int,
        temperature: float=BODY_TEMP_KELVIN,
        temp_unit: Literal["K", "degC"] = "K"  # what, I THOUGHT THIS WAS AMERICA??
        ) -> Quantity[float]:
    """Computes the Nernst potential, RT/Fz * ln(c_out/c_in).
    Returns the potential as a pint quantity with millivolt unit"""
    
    conc = math.log(c_out / c_in)
    T = constants.Q_(temperature, temp_unit).to(constants.ureg.kelvin)

    res = conc * (constants.R * T) / (constants.F * float(z))

    return res.to(constants.ureg.millivolts)


# TODO add GHK equation. Rename module to 'membrane'. Maybe add dataclasses here or separately