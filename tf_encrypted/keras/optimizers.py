"""TFE Keras optimizers"""
from tensorflow.keras import backend as K

import tf_encrypted as tfe

class SGD():
  def __init__(self, lr=0.01):
    self.lr = lr

  def apply_gradients(self, var, grad):
    sess = K.get_session()
    for i, w in enumerate(var):
      sess.run(tfe.assign(w, w - grad[i] * self.lr))


def get(identifier):
  if isinstance(identifier, SGD):
    return identifier
  if isinstance(identifier, str):
    optimizers = {'sgd': SGD()}
    return optimizers[identifier]

  raise ValueError('Could not interpret '
                   'optimizer function identifier:', identifier)
