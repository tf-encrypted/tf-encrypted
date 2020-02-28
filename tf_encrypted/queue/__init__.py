"""
Queue data structures.
"""

from __future__ import absolute_import

from .fifo import AbstractFIFOQueue
from .fifo import FIFOQueue

__all__ = [
    "FIFOQueue",
    "AbstractFIFOQueue",
]
