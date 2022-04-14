"""Base abstraction for a Protocol."""
import functools
from abc import ABC
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Optional

import tensorflow as tf
import tf_encrypted as tfe

from ..tensor.factory import AbstractTensor

nodes = dict()


class Protocol(ABC):
    """
  Protocol is the base class that other protocols in TF Encrypted will extend.

  Do not directly instantiate this class.  You should use a subclass instead,
  such as :class:`~tf_encrypted.protocol.protocol.SecureNN`
  or :class:`~tf_encrypted.protocol.protocol.Pond`
  """

    def __enter__(self) -> "Protocol":
        self.last_protocol = tfe.get_protocol()
        tfe.set_protocol(self)
        return self

    def __exit__(
        self,
        exception_type,
        exception_value: Optional[Exception],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        tfe.set_protocol(self.last_protocol)

    def reset(self):
        pass


class TFETensor(ABC):
    pass

class TFEVariable(TFETensor):
    pass

class TFEPrivateTensor(TFETensor):
    pass

class TFEPrivateVariable(TFEPrivateTensor, TFEVariable):
    pass

class TFEPublicTensor(TFETensor):
    pass

class TFEPublicVariable(TFEPublicTensor, TFEVariable):
    pass

def make_hashable(x):
    if isinstance(x, (tuple, list)):
        return tuple([make_hashable(y) for y in x])
    elif isinstance(x, dict):
        return tuple(sorted([(make_hashable(item[0]), make_hashable(item[1])) for item in x.items()]))
    elif isinstance(x, tf.TensorShape):
        return tuple(x.as_list())
    else:
        try:
            hash(x)
            return x
        except TypeError:
            return id(x)

def memoize(func: Callable) -> Callable:
    """
  memoize(func) -> Callable

  Decorates a function for memoization, which explicitly caches the function's
  output.

  :param Callable func: The function to memoize
  """

    @functools.wraps(func)
    def cache_nodes(self: Protocol, *args: Any, **kwargs: Any) -> AbstractTensor:
        hashable_args = make_hashable(args)
        hashable_kwargs = make_hashable(kwargs)
        node_key = (func.__name__, hashable_args, hashable_kwargs)

        cached_result = nodes.get(node_key, None)
        if cached_result is not None:
            return cached_result

        result = func(self, *args, **kwargs)

        nodes[node_key] = result
        return result

    return cache_nodes


