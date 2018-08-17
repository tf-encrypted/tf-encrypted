import numpy as np
import math
from . import core

from typing import Any, Optional, Tuple, Union
from tensorflow_encrypted.protocol.pond import PondPrivateTensor


class Batchnorm(core.Layer):
    def __init__(self, mean: np.ndarray, variance: np.ndarray, scale: np.ndarray,
                 offset: np.ndarray, variance_epsilon: float = 1e-8) -> None:
        self.mean = mean
        self.variance = variance
        self.scale = scale
        self.offset = offset
        self.variance_epsilon = variance_epsilon
        self.denom = None

    def initialize(self, input_shape: Union(Tuple[int, int], Tuple[int, int, int, int])) -> None:
        # Batchnorm after Dense layer
        if len(input_shape) == 2:
            N, D = input_shape
            self.mean = self.mean.reshape(1, N)
            self.variance = self.variance.reshape(1, N)
            self.scale = self.scale.reshape(1, N)
            self.offset = self.offset.reshape(1, N)

        # Batchnorm after Conv2D layer
        elif len(input_shape) == 4:
            N, C, H, W = input_shape
            self.mean = self.mean.reshape(1, C, 1, 1)
            self.variance = self.variance.reshape(1, C, 1, 1)
            self.scale = self.scale.reshape(1, C, 1, 1)
            self.offset = self.offset.reshape(1, C, 1, 1)

        denomtemp = 1.0 / np.sqrt(self.variance + self.variance_epsilon)

        self.denom = self.prot.define_public_variable(denomtemp)
        self.mean = self.prot.define_public_variable(self.mean)
        self.variance = self.prot.define_public_variable(self.variance)
        self.scale = self.prot.define_public_variable(self.scale)
        self.offset = self.prot.define_public_variable(self.offset)

    def forward(self, x: 'PondPrivateTensor') -> 'PondPrivateTensor':
        if self.scale is None and self.offset is None:
            out = (x - self.mean) * self.denom
        elif self.scale is None:
            out = self.scale * (x - self.mean) * self.denom
        elif self.offset is None:
            out = (x - self.mean) * self.denom + self.offset
        else:
            out = self.scale * (x - self.mean) * self.denom + self.offset
        return out

    def backward(self) -> None:
        raise NotImplementedError
