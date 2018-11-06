import numpy as np
from . import core

from typing import List
from tf_encrypted.protocol.pond import PondPrivateTensor


class Batchnorm(core.Layer):

    """
    Batch Normalization Layer

    :param List[int] input_shape: input shape of the data flowing into the layer
    :param np.ndarray mean: ...
    :param np.ndarray variance: ...
    :param np.ndarray scale: ...
    :param np.ndarray offset: ...
    :param float variance_epsilon: ...
    """

    def __init__(self, input_shape: List[int],
                 mean: np.ndarray, variance: np.ndarray, scale: np.ndarray,
                 offset: np.ndarray, variance_epsilon: float = 1e-8) -> None:
        self.mean = mean
        self.variance = variance
        self.scale = scale
        self.offset = offset
        self.variance_epsilon = variance_epsilon
        self.denom = None

        super(Batchnorm, self).__init__(input_shape)

    def get_output_shape(self) -> List[int]:
        return self.input_shape

    def initialize(self) -> None:
        # Batchnorm after Dense layer
        if len(self.input_shape) == 2:
            N, D = self.input_shape
            self.mean = self.mean.reshape(1, N)
            self.variance = self.variance.reshape(1, N)
            self.scale = self.scale.reshape(1, N)
            self.offset = self.offset.reshape(1, N)

        # Batchnorm after Conv2D layer
        elif len(self.input_shape) == 4:
            N, C, H, W = self.input_shape
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

    def forward(self, x: PondPrivateTensor) -> PondPrivateTensor:
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
        """
        `backward` is not implemented for `batchnorm`

        :raises: NotImplementedError
        """
        raise NotImplementedError
