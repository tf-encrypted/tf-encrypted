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

    Reference: https://paperswithcode.com/method/sgd-with-momentum
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


class AMSgrad:
    """
    M = b1 * M + (1-b1) * G
    V = b2 * V + (1-b2) * G^2
    V_hat = max(V_hat, V)
    W = W - r * M / sqrt(V_hat)

    Reference: https://paperswithcode.com/method/amsgrad
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.Ms = {}
        self.Vs = {}
        self.Vhats = {}
        self.epsilon = 2**-tfe.get_protocol().fixedpoint_config.precision_fractional

    def compile(self, layers):
        for layer in layers:
            W = layer.weights
            self.Vs[id(W)] = [tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))), apply_scaling=True, share_type=W[i].share_type, factory=W[i].backing_dtype)
                    for i in range(len(W))]
            self.Ms[id(W)] = [tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))), apply_scaling=True, share_type=W[i].share_type, factory=W[i].backing_dtype)
                    for i in range(len(W))]
            self.Vhats[id(W)] = [tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))), apply_scaling=True, share_type=W[i].share_type, factory=W[i].backing_dtype)
                    for i in range(len(W))]

    def apply_gradients(self, W, G):
        if id(W) not in self.Vs:
            raise RuntimeError("Unregonized layer weights")

        M = self.Ms[id(W)]
        V = self.Vs[id(W)]
        Vhat = self.Vhats[id(W)]
        ops = []
        for i in range(len(W)):
            op1 = tfe.assign(M[i], self.beta1 * M[i] + (1-self.beta1) * G[i])
            op2 = tfe.assign(V[i], self.beta2 * V[i] + (1-self.beta2) * G[i] * G[i])
            with tf.control_dependencies([op2]):
                op3 = tfe.assign(Vhat[i], tfe.maximum(Vhat[i], V[i]))
            with tf.control_dependencies([op1, op3]):
                diff = self.learning_rate * M[i] * tfe.inv_sqrt(Vhat[i] + self.epsilon)
                ops.append(tfe.assign(W[i], W[i] - diff))
        return tf.group(*ops)


class Adam:
    """
    M = b1 * M + (1-b1) * G
    V = b2 * V + (1-b2) * G^2
    V_hat = max(V_hat, V)
    W = W - r * M / sqrt(V_hat)

    https://paperswithcode.com/method/adam
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.Ms = {}
        self.Vs = {}
        self.beta1_pow = {}
        self.beta2_pow = {}
        self.epsilon = 2**-tfe.get_protocol().fixedpoint_config.precision_fractional

    def compile(self, layers):
        for layer in layers:
            W = layer.weights
            self.Vs[id(W)] = [tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))), apply_scaling=True, share_type=W[i].share_type, factory=W[i].backing_dtype)
                    for i in range(len(W))]
            self.Ms[id(W)] = [tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))), apply_scaling=True, share_type=W[i].share_type, factory=W[i].backing_dtype)
                    for i in range(len(W))]
            factory = W[0].backing_dtype if len(W) > 0 else None
            self.beta1_pow[id(W)] = tfe.define_public_variable(np.array(1), apply_scaling=True, factory=factory)
            self.beta2_pow[id(W)] = tfe.define_public_variable(np.array(1), apply_scaling=True, factory=factory)

    def apply_gradients(self, W, G):
        if id(W) not in self.Vs:
            raise RuntimeError("Unregonized layer weights")

        beta1_pow = self.beta1_pow[id(W)]
        beta2_pow = self.beta2_pow[id(W)]
        beta1_op = tfe.assign(beta1_pow, beta1_pow * self.beta1)
        beta2_op = tfe.assign(beta2_pow, beta2_pow * self.beta2)

        M = self.Ms[id(W)]
        V = self.Vs[id(W)]
        ops = []
        for i in range(len(W)):
            op1 = tfe.assign(M[i], self.beta1 * M[i] + (1-self.beta1) * G[i])
            op2 = tfe.assign(V[i], self.beta2 * V[i] + (1-self.beta2) * G[i] * G[i])
            with tf.control_dependencies([op1, beta1_op]):
                mhat = M[i] / (1 - beta1_pow)
            with tf.control_dependencies([op2, beta2_op]):
                vhat = V[i] / (1 - beta2_pow)

            diff = self.learning_rate * mhat * tfe.inv_sqrt(vhat + self.epsilon)
            ops.append(tfe.assign(W[i], W[i] - diff))
        return tf.group(*ops)


_known_optimizers = {
    "sgd": SGD,
    "sgd_with_momentum": SGDWithMomentum,
    "amsgrad": AMSgrad,
    "adam": Adam,
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
