from __future__ import absolute_import
from typing import Optional, Any
import inspect
import tensorflow as tf

from .config import Config, LocalConfig, RemoteConfig, get_config
from .player import player
from .protocol import global_caches_updater, Pond, get_protocol
from .session import Session, setTFEDebugFlag, setMonitorStatsFlag, setTFETraceFlag
from . import convert
from . import layers
from . import operations
from . import protocol


_all_prot_funcs = protocol.get_all_funcs()


def _prot_func_not_implemented(*args: Any, **kwargs: Any) -> None:
    raise Exception(
        "This function is not implemented in protocol {}".format(inspect.stack()[1][3])
    )


def set_protocol(prot: Optional[protocol.Protocol] = None) -> None:
    """
    Sets the global protocol. See :class:`~tf_encrypted.protocol.protocol.Protocol` for
    more info.

    :param ~tf_encrypted.protocol.protocol.Protocol prot: A protocol instance.
    """

    # reset all names
    for func_name in _all_prot_funcs:
        globals()[func_name] = _prot_func_not_implemented

    # add global names according to new protocol
    if prot is not None:
        methods = inspect.getmembers(prot, predicate=inspect.ismethod)
        public_methods = [method for method in methods if not method[0].startswith('_')]
        for name, func in public_methods:
            globals()[name] = func

    # record new protocol
    protocol.set_protocol(prot)


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
    "operations",
    "global_caches_updater",
    "global_variables_initializer",
]
