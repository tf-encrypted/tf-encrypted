"""TFE Keras optimizers"""
import tensorflow as tf

import tf_encrypted as tfe

class SGD():
  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate

  def apply_gradients(self, var, grad):
    sess = tf.get_default_session()
    for i, w in enumerate(var):
      sess.run(tfe.assign(w, w + (grad[i] * self.learning_rate).negative()))
