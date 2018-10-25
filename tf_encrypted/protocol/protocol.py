from abc import ABC, abstractmethod
import functools
from typing import Optional, Any, Callable
from types import TracebackType

import tensorflow as tf

from ..tensor.factory import AbstractTensor


__PROTOCOL__ = None
global_cache_updators = list()
nodes = dict()


class Protocol(ABC):
    """
    Protocol is the base class that other protocols in tf-encrypted will extend from.

    Do not directly instantiate this class.  You should use a subclass instead, such as :class:`~tf_encrypted.protocol.protocol.SecureNN`
    or :class:`~tf_encrypted.protocol.protocol.Pond`
    """

    def __enter__(self) -> 'Protocol':
        set_protocol(self)
        return self

    def __exit__(self, type,  # type is `Optional[Type[BaseException]]`, but declaring `Type` breaks readthedocs.
                 value: Optional[Exception],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        set_protocol(None)
        return None

    @property
    @abstractmethod
    def initializer(self) -> tf.Operation:
        pass


def set_protocol(prot: Optional[Protocol]) -> None:
    """
    Sets the global protocol.  E.g. :class:`~tensorflow_encrypted.protocol.securenn.SecureNN`
    or :class:`~tensorflow_encrypted.protocol.pond.Pond`.

    .. code-block::python
        tfe.set_protocol(tfe.protocol.secureNN())

    :param ~tensorflow_encrypted.protocol.protocol.Protocol prot: An instance of a tfe protocol.
    """
    global __PROTOCOL__
    __PROTOCOL__ = prot


def get_protocol() -> Optional[Protocol]:
    """
    :rtype: ~tensorflow_encrypted.protocol.protocol.Protocol
    :returns: The global protocol.
    """
    return __PROTOCOL__


def global_caches_updator() -> tf.Operation:
    with tf.name_scope('cache_update'):
        return tf.group(*global_cache_updators)


def memoize(func: Callable) -> Callable:

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
