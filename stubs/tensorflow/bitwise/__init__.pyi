from typing import Union, Optional

import numpy as np

from .. import Tensor


def bitwise_and(
    x: Union[Tensor, np.ndarray],
    y: Union[Tensor, np.ndarray, int],
    name: Optional[str]=None
) -> Tensor:
    ...


def right_shift(
    x: Union[Tensor, np.ndarray],
    y: Union[Tensor, np.ndarray],
    name: Optional[str]=None
) -> Tensor:
    ...
