import numpy as np
import math
from . import core


class Batchnorm(core.Layer):
    def __init__(self, mean, var, gamma, beta, epsilone=1e-8, momentum=0.99):
        self.epsilone = epsilone
        self.momentum = momentum
        self.mean = mean
        self.var = var
        self.gamma = gamma
        self.beta = beta
        self.denom = None

    def initialize(self, input_shape, initial_weights=None):
        if len(input_shape) == 2:
            N, D = input_shape
            self.mean = self.mean.reshape(1, N)
            self.var = self.var.reshape(1, N)
            self.gamma = self.gamma.reshape(1, N)
            self.beta = self.beta.reshape(1, N)

        elif len(input_shape) == 4:
            N, C, H, W = input_shape
            self.mean = self.mean.reshape(1, C, 1, 1)
            self.var = self.var.reshape(1, C, 1, 1)
            self.gamma = self.gamma.reshape(1, C, 1, 1)
            self.beta = self.beta.reshape(1, C, 1, 1)

        denomtemp = 1.0 / np.sqrt(self.var + 1e-8)

        self.mean = self.prot.define_public_variable(self.mean)
        self.var = self.prot.define_public_variable(self.var)
        self.gamma = self.prot.define_public_variable(self.gamma)
        self.beta = self.prot.define_public_variable(self.beta)

        #denomtemp = 1.0 / np.sqrt(self.var + 1e-8)
        self.denom = self.prot.define_public_variable(denomtemp)

    def forward(self, x):
        out = self.gamma * (x - self.mean) * self.denom + self.beta
        return out

    def backward(self):
        pass
