"""
Queue data structures.
"""

from __future__ import absolute_import

from .fifo import AbstractFIFOQueue
from .fifo import FIFOQueue
from .fifo import TFFIFOQueue

__all__ = [
    "TFFIFOQueue",
    "FIFOQueue",
    "AbstractFIFOQueue",
]
