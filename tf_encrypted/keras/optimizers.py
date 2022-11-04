"""TFE Keras optimizers"""
import numpy as np
import tensorflow as tf

import tf_encrypted as tfe


class SGD:
    """
    Stochastic gradient descent optimizer.

    V = bV + aG
    W = W - V

    Reference: https://paperswithcode.com/method/sgd-with-momentum
    """

    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.Vs = {}

    def compile(self, weights):
        for W in weights:
            self.Vs[id(W)] = [
                tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))),
                    apply_scaling=True,
                    share_type=W[i].share_type,
                    factory=W[i].backing_dtype,
                )
                for i in range(len(W))
            ]

    def apply_gradients(self, W, G):
        if id(W) not in self.Vs:
            raise RuntimeError("Unregonized layer weights")

        with tf.name_scope("SGD-apply-gradients"):
            V = self.Vs[id(W)]
            for i in range(len(W)):
                diff = self.momentum * V[i].read_value()
                tfe.assign(W[i], W[i].read_value() - (diff + self.learning_rate * G[i]))
                tfe.assign(V[i], diff + self.learning_rate * G[i])


class AMSgrad:
    """
    M = b1 * M + (1-b1) * G
    V = b2 * V + (1-b2) * G^2
    V_hat = max(V_hat, V)
    W = W - r * M / sqrt(V_hat)

    Reference: https://paperswithcode.com/method/amsgrad
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.Ms = {}
        self.Vs = {}
        self.Vhats = {}
        self.epsilon = 2 ** -tfe.get_protocol().fixedpoint_config.precision_fractional

    def compile(self, weights):
        for W in weights:
            self.Vs[id(W)] = [
                tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))),
                    apply_scaling=True,
                    share_type=W[i].share_type,
                    factory=W[i].backing_dtype,
                )
                for i in range(len(W))
            ]
            self.Ms[id(W)] = [
                tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))),
                    apply_scaling=True,
                    share_type=W[i].share_type,
                    factory=W[i].backing_dtype,
                )
                for i in range(len(W))
            ]
            self.Vhats[id(W)] = [
                tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))),
                    apply_scaling=True,
                    share_type=W[i].share_type,
                    factory=W[i].backing_dtype,
                )
                for i in range(len(W))
            ]

    def apply_gradients(self, W, G):
        if id(W) not in self.Vs:
            raise RuntimeError("Unregonized layer weights")

        with tf.name_scope("AMSgard-apply-gradients"):
            M = self.Ms[id(W)]
            V = self.Vs[id(W)]
            Vhat = self.Vhats[id(W)]
            for i in range(len(W)):
                tfe.assign(
                    M[i], self.beta1 * M[i].read_value() + (1 - self.beta1) * G[i]
                )
                tfe.assign(
                    V[i],
                    self.beta2 * V[i].read_value() + (1 - self.beta2) * G[i] * G[i],
                )
                v_max = tfe.maximum(Vhat[i].read_value(), V[i].read_value())
                tfe.assign(Vhat[i], v_max)
                vhat_add_e = Vhat[i].read_value() + self.epsilon
                diff = self.learning_rate * M[i].read_value() * tfe.inv_sqrt(vhat_add_e)
                tfe.assign(W[i], W[i].read_value() - diff)


class Adam:
    """
    M = b1 * M + (1-b1) * G
    V = b2 * V + (1-b2) * G^2
    M_hat = M / (1 - beta1**t)
    V_hat = V / (1 - beta2**t)
    W = W - r * M_hat / sqrt(V_hat)

    https://paperswithcode.com/method/adam
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.Ms = {}
        self.Vs = {}
        self.beta1_pow = {}
        self.beta2_pow = {}
        self.epsilon = 2 ** -tfe.get_protocol().fixedpoint_config.precision_fractional

    def compile(self, weights):
        for W in weights:
            self.Vs[id(W)] = [
                tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))),
                    apply_scaling=True,
                    share_type=W[i].share_type,
                    factory=W[i].backing_dtype,
                )
                for i in range(len(W))
            ]
            self.Ms[id(W)] = [
                tfe.define_private_variable(
                    np.zeros(list(map(int, W[i].shape.as_list()))),
                    apply_scaling=True,
                    share_type=W[i].share_type,
                    factory=W[i].backing_dtype,
                )
                for i in range(len(W))
            ]
            factory = W[0].backing_dtype if len(W) > 0 else None
            self.beta1_pow[id(W)] = tfe.define_public_variable(
                np.array(1), apply_scaling=True, factory=factory
            )
            self.beta2_pow[id(W)] = tfe.define_public_variable(
                np.array(1), apply_scaling=True, factory=factory
            )

    def apply_gradients(self, W, G):
        if id(W) not in self.Vs:
            raise RuntimeError("Unregonized layer weights")

        with tf.name_scope("adam-apply-gradients"):
            beta1_pow = self.beta1_pow[id(W)]
            beta2_pow = self.beta2_pow[id(W)]
            tfe.assign(beta1_pow, beta1_pow.read_value() * self.beta1)
            tfe.assign(beta2_pow, beta2_pow.read_value() * self.beta2)

            M = self.Ms[id(W)]
            V = self.Vs[id(W)]
            for i in range(len(W)):
                tfe.assign(
                    M[i], self.beta1 * M[i].read_value() + (1 - self.beta1) * G[i]
                )
                tfe.assign(
                    V[i],
                    self.beta2 * V[i].read_value() + (1 - self.beta2) * G[i] * G[i],
                )
                mhat = M[i].read_value() / (1 - beta1_pow.read_value())
                vhat = V[i].read_value() / (1 - beta2_pow.read_value())
                diff = self.learning_rate * mhat * tfe.inv_sqrt(vhat + self.epsilon)
                tfe.assign(W[i], W[i].read_value() - diff)


_known_optimizers = {
    "sgd": SGD,
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
