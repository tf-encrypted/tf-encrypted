"""TFE Keras backend"""
import threading

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.backend import get_graph, name_scope

import tf_encrypted as tfe

# This is a thread local object that will hold the default internal TFE session
# used by TFE Keras. It can be set manually via `set_session(sess)`.
_SESSION = threading.local()


def get_session(op_input_list=()):
  """Returns the session object for the current thread."""
  global _SESSION
  default_session = ops.get_default_session()
  if default_session is not None:
    session = default_session
    # If the default session is not a TFE Session, create one
    if not isinstance(session, tfe.Session()):
      session = tfe.Session()
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
  graph = get_graph()
  with graph.as_default():
    with name_scope(''):
      array_ops.placeholder_with_default(
          False, shape=(), name='keras_placeholder')
