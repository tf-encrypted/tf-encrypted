from __future__ import absolute_import
import inspect

from .pond import Pond, TFEVariable, TFETensor
from .securenn import SecureNN

from .protocol import Protocol, global_caches_updator, memoize, set_protocol, get_protocol


def get_all_funcs() -> list:
    all_prot_methods = []
    all_prot_method_names = []

    protocols = [Pond(), SecureNN()]

    for protocol in protocols:
        methods = inspect.getmembers(Pond(), predicate=inspect.ismethod)
        for method in methods:
            if method[0] not in all_prot_method_names and method[0][0] is not '_':
                all_prot_method_names.append(method[0])
                all_prot_methods.append(method)

    return all_prot_methods


__all__ = [
    'Protocol',
    'global_caches_updator',
    'memoize',
    'Pond',
    'SecureNN',
    'TFEVariable',
    'TFETensor',
    'set_protocol',
    'get_protocol'
]
