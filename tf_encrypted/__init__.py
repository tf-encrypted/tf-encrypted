"""TF Encrypted namespace."""
from __future__ import absolute_import
from functools import wraps
from typing import Optional, Any
import inspect

import tensorflow as tf

from .config import Config
from .config import LocalConfig
from .config import RemoteConfig
from .config import get_config
from .config import EagerLocalConfig
from .player import player
from .protocol import get_protocol
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


_all_prot_funcs = protocol.get_all_funcs()


def _prot_func_not_implemented(*args: Any, **kwargs: Any) -> None:
  msg = "This function is not implemented in protocol {}"
  raise Exception(msg.format(inspect.stack()[1][3]))


def local_computation(
    player_name=None,
    **kwargs
):
  """Annotate a function `compute_func` for local computation.

  This decorator can be used to pin a function's code to a specific player's
  device for remote execution.  This is useful when defining player-specific
  handlers for e.g. providing model weights, or input and output tensors.

  The decorator can handle global functions, normal object methods, or
  classmethods. If wrapping a method, it's presumed that the method's object
  has an attribute named `player_name`, or that the user will provide the
  `player_name` later on as a kwarg to the `compute_func`.

  Example:
    ```
    @tfe.local_computation('input-provider')
    def provide_input():
      return tf.random.normal((3, 3))

    @tfe.local_computation
    def receive_output(logits):
      return tf.print(tf.argmax(logits, axis=-1))

    x = provide_input()
    y = model(x)
    receive_op = receive_output(y, player_name='output-receiver')
    with tfe.Session():
      sess.run(receive_op)
    ```

  Arguments:
    player_name: Name of the player who should execute the function.
    kwargs: Keyword arguments to use when encoding or encrypting
      inputs/outputs to compute_func: see tfe.define_local_computation for
      details.

  Returns:
    The compute_func, but decorated for remote execution.
  """
  if callable(player_name):
    # The user has called us as a standard decorator:
    #
    # @tfe.local_computation
    # def provide_input():
    #   return tf.zeros((2, 2))
    actual_compute_func = player_name
    player_name = None
  else:
    # The user has called us as a function, maybe with non-default args:
    #
    # @tfe.local_computation('input-provider')
    # def provide_input():
    #   return tf.zeros((2, 2))
    actual_compute_func = None

  def decorator(compute_func):

    @wraps(compute_func)
    def compute_func_wrapper(*compute_func_args, **compute_func_kwargs):

      # Assumer user has passed player_name to decorator. If not, try to recover.
      actual_player_name = player_name
      if actual_player_name is None:
        # Maybe user has passed player_name to compute_func as a kwarg
        actual_player_name = compute_func_kwargs.get("player_name", None)
      if actual_player_name is None:
        # Assume compute_func is a method and its instance has some attribute
        # 'player_name'
        if compute_func_args:
          parent_instance = compute_func_args[0]
          actual_player_name = getattr(parent_instance, 'player_name', None)
      if actual_player_name is None:
        # Fallback to error
        raise ValueError("'player_name' not provided. Please provide "
                         "'player_name' as a keyword argument to this "
                         "function, or as an argument to the "
                         "tfe.local_computation decorator.")

      return get_protocol().define_local_computation(
          actual_player_name,
          compute_func,
          arguments=compute_func_args,
          **kwargs,
      )

    return compute_func_wrapper

  if actual_compute_func is None:
    # User has not yet passed a compute_func, so we'll expect them to
    # pass it outside of this function's scope (e.g. as a decorator).
    return decorator

  # User has already passed a compute_func, so return the decorated version.
  return decorator(actual_compute_func)


def set_protocol(prot: Optional[protocol.Protocol] = None) -> None:
  """
  Sets the global protocol. See
  :class:`~tf_encrypted.protocol.protocol.Protocol` for more info.

  :param ~tf_encrypted.protocol.protocol.Protocol prot: A protocol instance.
  """

  # reset all names
  for func_name in _all_prot_funcs:
    globals()[func_name] = _prot_func_not_implemented

  # add global names according to new protocol
  if prot is not None:
    methods = inspect.getmembers(prot, predicate=inspect.ismethod)
    public_methods = [
        method for method in methods if not method[0].startswith('_')]
    for name, func in public_methods:
      globals()[name] = func

  # record new protocol
  protocol.set_protocol(prot)


def set_config(config: Config) -> None:
  from .config import set_config as set_global_config

  set_global_config(config)
  set_protocol(None)


def global_variables_initializer() -> Optional[tf.Operation]:
  prot = protocol.get_protocol()
  if prot is not None:
    return prot.initializer
  return None


set_protocol(Pond())


__all__ = [
    "LocalConfig",
    "RemoteConfig",
    "EagerLocalConfig",
    "set_tfe_events_flag",
    "set_tfe_trace_flag",
    "set_log_directory",
    "get_config",
    "set_config",
    "get_protocol",
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
