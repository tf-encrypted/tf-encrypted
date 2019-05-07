"""Module containing implementations of secure protocols."""

from __future__ import absolute_import
import inspect

from .protocol import (
    Protocol,
    memoize,
    set_protocol,
    get_protocol,
    nodes,
)

from .pond import Pond, TFEVariable, TFETensor
from .securenn import SecureNN


def get_all_funcs():
  """Assemble public method names from all protocols into a list."""
  all_prot_method_names = set()

  protocols = [Pond, SecureNN]
  for protocol in protocols:
    members = inspect.getmembers(protocol, predicate=inspect.isfunction)
    all_prot_method_names |= set(
        func_name
        for func_name, _ in members
        if not func_name.startswith('_')
    )

  return all_prot_method_names


__all__ = [
    "Protocol",
    "memoize",
    "Pond",
    "SecureNN",
    "TFEVariable",
    "TFETensor",
    "set_protocol",
    "get_protocol",
]
