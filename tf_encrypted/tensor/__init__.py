"""Tensors representing non-native data types (like fixed-point precision)."""
from __future__ import absolute_import

import tensorflow as tf

from .boolfactory import bool_factory
from .fixed import _validate_fixedpoint_config
from .fixed import fixed64
from .fixed import fixed64_heuristic
from .fixed import fixed64_ni
from .fixed import fixed100
from .fixed import fixed100_ni
from .int100 import int100factory
from .native import native_factory

int1factory = bool_factory()
int8factory = native_factory(tf.int8)
int16factory = native_factory(tf.int16)
int32factory = native_factory(tf.int32)
int64factory = native_factory(tf.int64)

factories = {
    0: int1factory,
    1: int8factory,
    2: int8factory,
    4: int8factory,
    8: int8factory,
    16: int16factory,
    32: int32factory,
    64: int64factory,
}
factories.update(
    {
        tf.bool: int1factory,
        tf.int8: int8factory,
        tf.int16: int16factory,
        tf.int32: int32factory,
        tf.int64: int64factory,
    }
)

assert _validate_fixedpoint_config(fixed100, int100factory)
assert _validate_fixedpoint_config(fixed100_ni, int100factory)
assert _validate_fixedpoint_config(fixed64, int64factory)
assert _validate_fixedpoint_config(fixed64_ni, int64factory)
assert _validate_fixedpoint_config(fixed64_heuristic, int64factory)

__all__ = [
    "int1factory",
    "int8factory",
    "int16factory",
    "int32factory",
    "int64factory",
    "int100factory",
    "factories",
]
