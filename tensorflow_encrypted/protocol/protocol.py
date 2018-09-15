import functools
from typing import Optional, Type, List, Dict
from types import TracebackType

import tensorflow as tf


_current_prot = None
_global_cache_updators: List = []
_nodes: Dict = {}


class Protocol(object):

    def __enter__(self) -> 'Protocol':
        set_protocol(self)
        return self

    def __exit__(self, type: Optional[Type[BaseException]],
                 value: Optional[Exception],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        set_protocol(None)
        return None


def set_protocol(prot: Optional[Protocol]) -> None:
    global _current_prot
    _current_prot = prot


def get_protocol() -> Optional[Protocol]:
    return _current_prot


def global_caches_updator():
    with tf.name_scope('cache_update'):
        return tf.group(*_global_cache_updators)


def memoize(func):

    @functools.wraps(func)
    def cache_nodes(self, *args, **kwargs):

        node_key = (func.__name__, args, tuple(sorted(kwargs.items())))

        cached_result = _nodes.get(node_key, None)
        if cached_result is not None:
            return cached_result

        result = func(self, *args, **kwargs)

        _nodes[node_key] = result
        return result

    return cache_nodes
