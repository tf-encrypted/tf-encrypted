from typing import Optional, Any, Type
from types import TracebackType
from __future__ import absolute_import

# from unencrypted_native import UnencryptedNative
# from unencrypted_fixedpoint import UnencryptedFixedpoint
from .pond import Pond
from .securenn import SecureNN
# from secureml import SecureML


_current_prot = None


def set_protocol(prot: Optional[Protocol]) -> None:
    global _current_prot
    _current_prot = prot


def get_protocol() -> Optional[Protocol]:
    return _current_prot


class Player(object):
    def __init__(self, device_name: str) -> None:
        self.device_name = device_name


class Protocol(object):
    def __enter__(self) -> Protocol:
        set_protocol(self)
        return self

    def __exit__(self, type: Optional[Type[BaseException]],
                 value: Optional[Exception],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        set_protocol(None)
        return None
