from typing import Any

import pint

# Create a registry for units
ureg = pint.UnitRegistry()


def _passthrough[T](cls_: T) -> T:
    """Setting Q_ as an alias of ureg.Quantity is recommended, but causes some type checking hiccups.
    This just explicates that whatever passes through this function retains its type, so
    we can use it to set the alias"""
    return cls_


# Alias for a quantity
Q_: type[pint.Quantity[Any]] = _passthrough(ureg.Quantity)

# Gas constant
R = Q_(8.314_462_618, ureg.joule / (ureg.mole * ureg.kelvin))

# Faraday constant
F = Q_(96_485.332_12, ureg.coulomb / ureg.mole)
