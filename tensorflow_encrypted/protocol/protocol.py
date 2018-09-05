from typing import Optional, Type, List
from types import TracebackType
import tensorflow as tf


_current_prot = None
_global_cache_updators: List = []


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
