"""Base abstraction for a Protocol."""
from abc import ABC, abstractmethod
import functools
from typing import Optional, Any, Callable
from types import TracebackType

import tensorflow as tf

from ..tensor.factory import AbstractTensor


__PROTOCOL__ = None  # pylint: disable=invalid-name
nodes = dict()


class Protocol(ABC):
  """
  Protocol is the base class that other protocols in TF Encrypted will extend.

  Do not directly instantiate this class.  You should use a subclass instead,
  such as :class:`~tf_encrypted.protocol.protocol.SecureNN`
  or :class:`~tf_encrypted.protocol.protocol.Pond`
  """

  def __enter__(self) -> "Protocol":
    self.last_protocol = get_protocol()
    set_protocol(self)
    return self

  def __exit__(
      self,
      # type is `Optional[Type[BaseException]]`, but declaring `Type` breaks readthedocs.
      exception_type,
      exception_value: Optional[Exception],
      traceback: Optional[TracebackType],
  ) -> Optional[bool]:
    set_protocol(self.last_protocol)

  @property
  @abstractmethod
  def initializer(self) -> tf.Operation:
    pass


def set_protocol(prot: Protocol) -> None:
  """
  set_protocol(prot)

  Sets the global protocol.
  E.g. :class:`~tf_encrypted.protocol.securenn.SecureNN` or
  :class:`~tf_encrypted.protocol.pond.Pond`.

  .. code-block::python
      tfe.set_protocol(tfe.protocol.secureNN())

  :param Protocol prot: An instance of a tfe protocol.
  """
  global __PROTOCOL__  # pylint: disable=invalid-name
  __PROTOCOL__ = prot


def get_protocol() -> Optional[Protocol]:
  """
  get_protocol() -> Protocol or None

  Returns the current global protocol.
  """
  return __PROTOCOL__


def memoize(func: Callable) -> Callable:
  """
  memoize(func) -> Callable

  Decorates a function for memoization, which explicitly caches the function's
  output.

  :param Callable func: The function to memoize
  """
  @functools.wraps(func)
  def cache_nodes(self: Protocol, *args: Any, **kwargs: Any) -> AbstractTensor:
    args = tuple(tuple(x) if isinstance(x, list) else x for x in args)
    node_key = (func.__name__, args, tuple(sorted(kwargs.items())))

    cached_result = nodes.get(node_key, None)
    if cached_result is not None:
      return cached_result

    result = func(self, *args, **kwargs)

    nodes[node_key] = result
    return result

  return cache_nodes
