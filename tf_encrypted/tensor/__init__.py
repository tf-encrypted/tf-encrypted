from __future__ import absolute_import

from .prime import PrimeFactory
from .int32 import int32factory
from .int64 import int64factory
from .int100 import int100factory

from .fixed import (
    _validate_fixedpoint_config,
    fixed100,
    fixed100_ni,
    fixed64,
    fixed64_ni,
)

assert _validate_fixedpoint_config(fixed100, int100factory)
assert _validate_fixedpoint_config(fixed100_ni, int100factory)
assert _validate_fixedpoint_config(fixed64, int64factory)
assert _validate_fixedpoint_config(fixed64_ni, int64factory)

__all__ = [
    'PrimeFactory',
    'int32factory',
    'int64factory',
    'int100factory',
]
