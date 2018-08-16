from __future__ import absolute_import

class Player(object):
    def __init__(self, device_name: str) -> None:
        self.device_name = device_name

from .tensor import *
from .config import LocalConfig, RemoteConfig
from . import io
from . import protocol
from . import estimator
from . import layer

# __all__ = [
#     'InputProvider',
#     'LocalConfig',
#     'RemoteConfig',
#     'estimator',
#     'layer',
#     'protocol'
# ]
