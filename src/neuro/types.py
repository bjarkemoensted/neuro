from collections.abc import Callable
from typing import Literal, TypeGuard, get_args

import numpy as np
import numpy.typing as npt

# The allowed types for matplotlib colors :'(
type ColorType = (
    tuple[float, float, float]
    | str
    | tuple[float, float, float, float]
    | tuple[tuple[float, float, float] | str, float]
    | tuple[tuple[float, float, float, float], float]
)


type FuncType = Callable[[float], float] | Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]


type FixedPointCategory = Literal["stable", "unstable", "saddle", "non-hyperbolic"]


def is_fixed_point_category(obj: object) -> TypeGuard[FixedPointCategory]:
    return obj in get_args(FixedPointCategory.__value__)
