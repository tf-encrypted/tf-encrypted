"""TF Encrypted namespace."""
from __future__ import absolute_import
from typing import Optional, Any
import inspect
import tensorflow as tf

from .config import Config
from .config import LocalConfig
from .config import RemoteConfig
from .config import get_config
from .player import player
from .protocol import Pond
from .session import Session
from .session import set_tfe_events_flag
from .session import set_tfe_trace_flag
from .session import set_log_directory
from . import convert
from . import layers
from . import operations
from . import protocol
from . import keras
from . import serving
from . import queue


__protocol__ = None
__all_prot_funcs__ = protocol.get_all_funcs()


def _prot_func_not_implemented(*args: Any, **kwargs: Any) -> None:
  msg = "This function is not implemented in protocol {}"
  raise Exception(msg.format(inspect.stack()[1][3]))


def _update_protocol(prot):
  """Update current protocol in scope."""
  global __protocol__
  __protocol__ = prot


def get_protocol():
  """Return the current protocol in scope.

  Note this should not be used for accessing public protocol methods, use
  tfe.<public_protocol_method> instead.
  """
  return __protocol__


def set_protocol(prot: Optional[protocol.Protocol] = None) -> None:
  """
  Sets the global protocol. See
  :class:`~tf_encrypted.protocol.protocol.Protocol` for more info.

  :param ~tf_encrypted.protocol.protocol.Protocol prot: A protocol instance.
  """

  # reset all names
  for func_name in __all_prot_funcs__:
    globals()[func_name] = _prot_func_not_implemented

  # add global names according to new protocol
  if prot is not None:
    methods = inspect.getmembers(prot, predicate=inspect.ismethod)
    public_methods = [
        method for method in methods if not method[0].startswith('_')]
    for name, func in public_methods:
      globals()[name] = func

  # record new protocol
  _update_protocol(prot)


def set_config(config: Config) -> None:
  from .config import set_config as set_global_config

  set_global_config(config)
  set_protocol(None)


def global_variables_initializer() -> tf.Operation:
  return tf.global_variables_initializer()


set_protocol(Pond())


__all__ = [
    "LocalConfig",
    "RemoteConfig",
    "set_tfe_events_flag",
    "set_tfe_trace_flag",
    "set_log_directory",
    "get_config",
    "set_config",
    "set_protocol",
    "Session",
    "player",
    "protocol",
    "layers",
    "convert",
    "operations",
    "global_variables_initializer",
    "keras",
    "queue",
    "serving",
]
