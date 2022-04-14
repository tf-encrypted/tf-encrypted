"""TFE Keras optimizers"""
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.keras import backend as KE


class SGD:
    """Stochastic gradient descent optimizer.

    Arguments:
        learning_rate: float >= 0. Learning rate.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def apply_gradients(self, var, grad):
        ops = []
        for i, w in enumerate(var):
            ops.append(tfe.assign(w, w - grad[i] * self.learning_rate))
        return tf.group(*ops)


_known_optimizers = {
    "sgd": SGD,
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
