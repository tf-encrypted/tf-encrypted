from . import core


class Reshape(core.Layer):
    def __init__(self, shape=[-1]):
        self.shape = shape
        self.layer_output = None

    def initialize(self, input_shape, initializer=None):
        pass

    def forward(self, x):
        y = self.prot.reshape(x, self.shape)
        self.layer_output = y
        return y

    def backward(self, d_y, *args):
        pass
