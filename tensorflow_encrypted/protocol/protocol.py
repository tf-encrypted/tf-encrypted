from abc import ABC, abstractmethod
import functools
from typing import Optional, Type, Any, Callable
from types import TracebackType

import tensorflow as tf

from ..tensor.factory import AbstractTensor


__PROTOCOL__ = None
global_cache_updators = list()
nodes = dict()


class Protocol(ABC):

    def __enter__(self) -> 'Protocol':
        set_protocol(self)
        return self

    def __exit__(self, type: Optional[Type[BaseException]],
                 value: Optional[Exception],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        set_protocol(None)
        return None

    @property
    @abstractmethod
    def initializer(self) -> tf.Operation:
        pass


def set_protocol(prot: Optional[Protocol]) -> None:
    global __PROTOCOL__
    __PROTOCOL__ = prot


def get_protocol() -> Optional[Protocol]:
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
