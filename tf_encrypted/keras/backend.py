"""TFE Keras backend"""
import tensorflow as tf
from tensorflow.keras import backend as K

import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondPrivateVariable


# TODO Should be moved outside backend, maybe in pond.py?
def is_variable_initialized(x):
  """Test if TFE private variable has been initialized"""
  assert isinstance(x, PondPrivateVariable), type(x)
  is_initialized = tf.is_variable_initialized(x.share0.variable)
  return is_initialized


def get_session():
  """Returns the TFE session to be used"""
  sess = K.get_session()
  # If session is not a tfe.Session, creates one
  # K.get_session() will return a tfe.Session if
  # TFE global session was previously set with set_session
  if not isinstance(sess, tfe.Session):
    sess = tfe.Session()

  return sess

def set_session(sess):
  """Sets the global TFE session."""
  return K.set_session(sess)

def clear_session():
  """Destroys the current TFE graph and creates a new one"""
  return K.clear_session()
