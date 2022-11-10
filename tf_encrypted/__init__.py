"""TF Encrypted namespace."""
from __future__ import absolute_import

import inspect
import os.path
from typing import Any
from typing import Optional

from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

from . import convert
from . import keras
from . import operations
from . import protocol
from . import queue
from . import serving
from .config import Config
from .config import LocalConfig
from .config import RemoteConfig
from .config import get_config
from .player import player
from .protocol import ABY3
from .protocol import function

try:
    _dist = get_distribution("tf_encrypted")
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, "tf_encrypted")):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = "Please install this project with setup.py"
else:
    __version__ = _dist.version

__protocol__ = None
__all_prot_funcs__ = protocol.get_all_funcs()


def _prot_func_not_implemented(*args: Any, **kwargs: Any) -> None:
    msg = "This function is not implemented in protocol {}"
    raise Exception(msg.format(inspect.stack()[1][3]))


def _update_protocol(prot):
    """Update current protocol in scope."""
    global __protocol__
    __protocol__ = prot


def get_protocol():
    """Return the current protocol in scope.

    Note this should not be used for accessing public protocol methods, use
    tfe.<public_protocol_method> instead.
    """
    return __protocol__


def set_protocol(prot: Optional[protocol.Protocol] = None) -> None:
    """Sets the global protocol.

    See :class:`~tf_encrypted.protocol.protocol.Protocol` for more info.

    :param ~tf_encrypted.protocol.protocol.Protocol prot: A protocol instance.
    """

    # reset all names
    for func_name in __all_prot_funcs__:
        globals()[func_name] = _prot_func_not_implemented

    # add global names according to new protocol
    if prot is not None:
        methods = inspect.getmembers(prot, predicate=inspect.ismethod)
        public_methods = [method for method in methods if not method[0].startswith("_")]
        for name, func in public_methods:
            globals()[name] = func

    # record new protocol
    _update_protocol(prot)


def set_config(config: Config) -> None:
    # pylint: disable=import-outside-toplevel
    from .config import set_config as set_global_config

    # pylint: enable=import-outside-toplevel

    set_global_config(config)
    set_protocol(None)
    # Reset the graph to clear all ops that were created under
    # previous config that might use different devices,
    # otherwise there might be invalid device error.
    reset_default_graph()


# from .protocol import Pond
# set_protocol(Pond())
set_protocol(ABY3())

__all__ = [
    "LocalConfig",
    "RemoteConfig",
    "set_tfe_events_flag",
    "set_tfe_trace_flag",
    "set_log_directory",
    "get_config",
    "set_config",
    "function",
    "set_protocol",
    "player",
    "primitives",
    "protocol",
    "convert",
    "operations",
    "keras",
    "queue",
    "serving",
]
