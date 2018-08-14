from . import core

class Sigmoid(core.Layer):
    def __init__(self):
        self.layer_output = None

    def initialize(self, input_shape, initializer=None):
        pass

    def forward(self, x):
        y = self.prot.sigmoid(x)
        self.layer_output = y
        return y

    def backward(self, d_y, *args):
        y = self.layer_output
        d_x = d_y * y * (y.neg() + 1)
        return d_x
