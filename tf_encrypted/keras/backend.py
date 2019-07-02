"""TFE Keras backend"""
from tensorflow.keras import backend as K

import tf_encrypted as tfe


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
