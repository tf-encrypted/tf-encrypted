from __future__ import absolute_import
from typing import Optional
import inspect
import tensorflow as tf

from .config import Config, LocalConfig, RemoteConfig, get_config
from .session import setTFEDebugFlag, setMonitorStatsFlag, Session
from .protocol import global_caches_updator, Pond, get_protocol
from .player import player
from . import io
from . import protocol
from . import layers
from . import convert


def get_protocol_public_func(prot: protocol.Protocol) -> list:
    methods = inspect.getmembers(prot, predicate=inspect.ismethod)
    public_prot_methods = [method for method in methods if method[0][0] is not '_']

    return public_prot_methods


def set_protocol(prot: Optional[protocol.Protocol] = None) -> None:
    previous_prot = get_protocol()
    if previous_prot is not None:
        funcs = get_protocol_public_func(previous_prot)
        for func in funcs:
            del globals()[func[0]]

    protocol.set_protocol(prot)
    if prot is not None:
        funcs = get_protocol_public_func(prot)

        for func in funcs:
            globals()[func[0]] = func[1]


def set_config(config: Config) -> None:
    from .config import set_config as set_global_config

    set_global_config(config)
    set_protocol(None)


def get_global_variables() -> Optional[tf.Operation]:
    prot = protocol.get_protocol()
    if prot is not None:
        return prot.initializer
    else:
        return None


set_protocol(Pond())

__all__ = [
    "LocalConfig",
    "RemoteConfig",
    "setTFEDebugFlag",
    "setMonitorStatsFlag",
    "get_config",
    "set_config",
    "get_protocol",
    "set_protocol",
    "Session",
    "io",
    "player",
    "protocol",
    "layers",
    "convert",
    "global_caches_updator",
    "get_global_variables",
]
