from typing import Optional, Type
from types import TracebackType


_current_prot = None


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
