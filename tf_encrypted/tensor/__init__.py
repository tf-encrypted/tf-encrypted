from __future__ import absolute_import

from .int100 import (
    int100factory,
    Int100Constant,
    Int100Placeholder,
    Int100Variable,
    Int100Tensor,
    Int100SeededTensor
)

from .prime import (
    PrimeTensor,
    PrimePlaceholder,
    PrimeVariable,
    PrimeConstant,
    PrimeFactory,
)

from .int32 import (
    Int32Tensor,
    Int32Placeholder,
    Int32Variable,
    Int32Constant,
    int32factory
)

from .int64 import int64factory

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
    'Int100Constant',
    'Int100Placeholder',
    'Int100Variable',
    'Int100Tensor',
    'Int100SeededTensor',
    'PrimeTensor',
    'PrimePlaceholder',
    'PrimeVariable',
    'PrimeConstant',
    'PrimeFactory',
    'int32factory',
    'Int32Tensor',
    'Int32Placeholder',
    'Int32Variable',
    'Int32Constant',
]
