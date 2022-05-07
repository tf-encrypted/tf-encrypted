"""TFE Keras optimizers"""
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.keras import backend as KE
import numpy as np


class SGD:
    """Stochastic gradient descent optimizer.
    W = W - aG

    Arguments:
        learning_rate: float >= 0. Learning rate.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def compile(self, layers):
        pass

    def apply_gradients(self, var, grad):
        ops = []
        for i, w in enumerate(var):
            ops.append(tfe.assign(w, w - grad[i] * self.learning_rate))
        return tf.group(*ops)


class SGDWithMomentum:
    """
    V = bV + aG
    W = W - V
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.Vs = {}

    def compile(self, layers):
        for layer in layers:
            W = layer.weights
            self.Vs[id(W)] = [tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))), apply_scaling=True, share_type=W[i].share_type, factory=W[i].backing_dtype)
                    for i in range(len(W))]

    def apply_gradients(self, W, G):
        if id(W) not in self.Vs:
            raise RuntimeError("Unregonized layer weights")

        V = self.Vs[id(W)]
        ops = []
        for i in range(len(W)):
            op = tfe.assign(V[i], self.momentum * V[i] + self.learning_rate * G[i])
            with tf.control_dependencies([op]):
                ops.append(tfe.assign(W[i], W[i] - V[i]))
        return tf.group(*ops)



_known_optimizers = {
    "sgd": SGD,
    "sgd_with_momentum": SGDWithMomentum,
}


def get(identifier):
    if isinstance(identifier, type):
        return identifier()
    if isinstance(identifier, str):
        global _known_optimizers
        optimizer = _known_optimizers.get(identifier, None)
        if optimizer is not None:
            return optimizer()
    return identifier
