from typing import Union

import tensorflow as tf
import numpy as np

from .pond import (
    PondPublicVariable,
    PondPrivateVariable,
    PondPublicTensor,
    PondPrivateTensor
)

TFEData = Union[np.ndarray, tf.Tensor]
TFEVariable = Union[PondPublicVariable, PondPrivateVariable, tf.Variable]
TFEPublicTensor = PondPublicTensor
TFETensor = Union[TFEPublicTensor, PondPrivateTensor]
