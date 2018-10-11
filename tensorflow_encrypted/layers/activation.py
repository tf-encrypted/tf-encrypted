from typing import List

from . import core


class Sigmoid(core.Layer):

    def get_output_shape(self) -> List[int]:
        return self.input_shape

    def initialize(self, *args, **kwargs) -> None:
        pass

    def forward(self, x):
        y = self.prot.sigmoid(x)
        self.layer_output = y
        return y

    def backward(self, d_y, *args):
        y = self.layer_output
        d_x = d_y * y * (y.neg() + 1)
        return d_x


class Relu(core.Layer):
    def get_output_shape(self) -> List[int]:
        return self.input_shape

    def initialize(self, *args, **kwargs) -> None:
        pass

    def forward(self, x):
        """
        :param ~tensorflow_encrypted.protocol.pond.PondTensor x: The input tensor
        :rtype: ~tensorflow_encrypted.protocol.pond.PondTensor
        :returns: A pond tensor with the same backing type as the input tensor.
        """
        y = self.prot.relu(x)
        self.layer_output = y
        return y

    # TODO Approximate Relu derivate to implement backward
    def backward(self, d_y, *args):
        raise NotImplementedError


class Tanh(core.Layer):

    def get_output_shape(self) -> List[int]:
        return self.input_shape

    def initialize(self, *args, **kwargs) -> None:
        pass

    def forward(self, x):
        y = self.prot.tanh(x)
        self.layer_output = y
        return y

    # TODO Approximate Relu derivate to implement backward
    def backward(self, d_y, *args):
        raise NotImplementedError
