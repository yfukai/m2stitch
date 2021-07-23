from typing import Any, Union

import numpy as np
import numpy.typing as npt

NumArray = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float_]

Int = Union[int,np.int_]
Float = Union[float,np.float_]