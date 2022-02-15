from typing import Any
from typing import Union

import numpy as np
import numpy.typing as npt

NumArray = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]
BoolArray = npt.NDArray[np.bool_]

Int = Union[int, np.int_]
Float = Union[float, np.float_]
