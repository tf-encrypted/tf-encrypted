"""TFE Keras backend.
Most of the code was borrowed from the tf.keras codebase.
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

    def valid_session(session):
        if session is None:
            return False
        if not isinstance(session, tfe.Session):
            return False
        if session.graph is not _current_graph(op_input_list):
            return False
        return True

    if ops.inside_function():
        raise RuntimeError("Cannot get session inside Tensorflow graph function.")

    # return any suitable session already specified
    session = getattr(_SESSION, "session", None)
    if valid_session(session):
        return session

    # return default TF session if of right type
    session = ops.get_default_session()
    if valid_session(session):
        return session

    # we don't have a suitable session, create and cache a new one
    _SESSION.session = tfe.Session()
    assert valid_session(_SESSION.session)
    return _SESSION.session


def _current_graph(op_input_list):
    """Return the graph members of `op_input_list`, or the current graph."""
    # pylint: disable=protected-access
    return ops._get_graph_from_inputs(op_input_list)


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
