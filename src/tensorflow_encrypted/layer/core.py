import numpy as np


class Layer:
    pass


class Dense(Layer):

    def __init__(self, num_nodes, num_inputs):
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs

        self.layer_input = None
        self.weights = None
        self.bias = None

    def initialize(self):
        #TODO: how to pass the current protcol? as a global var? or per layer?
        initial_weights = np.random.normal(scale=0.1, size=(self.num_inputs, self.num_nodes))
        self.weights = prot.define_private_variable(initial_weights)
        self.bias = prot.define_private_variable(np.zeros(1, self.num_nodes))

    def forward(self, x):
        self.layer_input = x
        y = x.dot(self.weights) + self.bias
        return y

    def backward(self, d_y, learning_rate):
        x = self.layer_input
        d_x = d_y.dot(self.weights.transpose())

        d_weights = x.transpose().dot(d_y)
        d_bias = d_y.sum(axis=0)

        self.weights.assign((d_weights * learning_rate).neg() + self.weights)
        self.bias.assign((d_bias * learning_rate).neg() + self.bias)

        return d_x


class Sigmoid(Layer):

    def __init__(self):
        self.layer_ouput = y
        pass

    def initialize(self, input_shape, initializer=None):
        pass

    def forward(self, x):
        #TODO[koen]: same story
        y = prot.sigmoid(x)
        self.layer_ouput = y
        return y

    def backward(self, d_y, learning_rate):
        y = self.layer_ouput
        d_x = d_y * y * (y.neg() + 1)
        return d_x
