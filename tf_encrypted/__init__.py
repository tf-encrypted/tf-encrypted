from __future__ import absolute_import
from typing import Optional, Any
import inspect
import tensorflow as tf

from .config import Config, LocalConfig, RemoteConfig, get_config
from .session import Session, setTFEDebugFlag, setMonitorStatsFlag, setTFETraceFlag
from .protocol import global_caches_updator, Pond, get_protocol
from .player import player
from . import protocol
from . import layers
from . import convert

all_prot_funcs = protocol.get_all_funcs()


def prot_func_not_implemented(*args: Any, **kwargs: Any) -> None:
    raise Exception("This function is not implemented in protocol {}".format(inspect.stack()[1][3]))


def get_protocol_public_func(prot: protocol.Protocol) -> list:
    methods = inspect.getmembers(prot, predicate=inspect.ismethod)
    public_prot_methods = [method for method in methods if method[0][0] is not '_']

    return public_prot_methods


def set_protocol(prot: Optional[protocol.Protocol] = None) -> None:
    """
    Sets the global protocol.  See :class:`~tensorflow_encrypted.protocol.protocol.Protocol` for more info.

    :param ~tensorflow_encrypted.protocol.protocol.Protocol prot: A protocol instance.
    """
    for func in all_prot_funcs:
        if func[0] in globals():
            del globals()[func[0]]

    protocol.set_protocol(prot)

    if prot is not None:
        funcs = get_protocol_public_func(prot)

        for func in funcs:
            globals()[func[0]] = func[1]

    for func in all_prot_funcs:
        if func[0] not in globals():
            globals()[func[0]] = prot_func_not_implemented


def set_config(config: Config) -> None:
    from .config import set_config as set_global_config

    set_global_config(config)
    set_protocol(None)


def global_variables_initializer() -> Optional[tf.Operation]:
    prot = protocol.get_protocol()
    if prot is not None:
        return prot.initializer
    else:
        return None


set_protocol(Pond())

__all__ = [
    "LocalConfig",
    "RemoteConfig",
    "setMonitorStatsFlag",
    "setTFETraceFlag",
    "setTFEDebugFlag",
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
    "global_variables_initializer",
]
