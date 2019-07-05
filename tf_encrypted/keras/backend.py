"""TFE Keras backend. Most of the code was
borrowed from the tf.keras codebase.
"""
import threading

from tensorflow.python.framework import ops

import tf_encrypted as tfe

# This is a thread local object that will hold the default internal TFE session
# used by TFE Keras. It can be set manually via `set_session(sess)`.
_SESSION = threading.local()


def get_session(op_input_list=()):
  """Returns the session object for the current thread."""
  global _SESSION
  if getattr(_SESSION, 'session', None) is not None:
    return _SESSION.session
  default_session = ops.get_default_session()
  if default_session is not None:
    # If the default session is a TFE Session return this session
    if isinstance(default_session, tfe.Session()):
      return default_session
    if not isinstance(default_session, tfe.Session()):
      raise TypeError(
          'The default session should be a tfe.Session(). '
          'You are probably trying to run this graph with '
          'tf.Session() instead of tfe.Session()')
  else:
    if ops.inside_function():
      raise RuntimeError('Cannot get session inside Tensorflow graph function.')
      # If we don't have a session, or that session does not match the current
      # graph, create and cache a new session.
    if (getattr(_SESSION, 'session', None) is None or
        _SESSION.session.graph is not _current_graph(op_input_list)):
      _SESSION.session = tfe.Session()
    session = _SESSION.session
  return session


def _current_graph(op_input_list):
  """Return the graph members of `op_input_list`, or the current graph."""
  return ops._get_graph_from_inputs(op_input_list) # pylint: disable=protected-access


def set_session(session):
  """Sets the global TFE session.
  Arguments:
      session: A TFE Session.
  """
  global _SESSION
  _SESSION.session = session


def clear_session():
  """Destroys the current TFE graph and creates a new one"""
  _SESSION.session = None
  ops.reset_default_graph()
