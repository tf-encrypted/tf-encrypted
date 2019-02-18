from __future__ import absolute_import
import inspect

from .pond import Pond, TFEVariable, TFETensor
from .securenn import SecureNN

from .protocol import (
    Protocol,
    global_caches_updater,
    memoize,
    set_protocol,
    get_protocol,
)


def get_all_funcs():
    all_prot_method_names = set()

    protocols = [Pond, SecureNN]
    for protocol in protocols:
        all_prot_method_names |= set(
            func_name
            for func_name, _ in inspect.getmembers(protocol, predicate=inspect.isfunction)
            if not func_name.startswith('_')
        )

    return all_prot_method_names


__all__ = [
    "Protocol",
    "global_caches_updater",
    "memoize",
    "Pond",
    "SecureNN",
    "TFEVariable",
    "TFETensor",
    "set_protocol",
    "get_protocol",
]
