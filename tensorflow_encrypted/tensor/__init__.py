from __future__ import absolute_import

from .int100 import (
    int100factory,
    Int100Constant,
    Int100Placeholder,
    Int100Variable,
    Int100Tensor
)

from .prime import (
    PrimeFactory,
    PrimeTensor,
    PrimePlaceholder,
    PrimeVariable,
    PrimeConstant
)

from .int32 import (
    int32factory,
    Int32Tensor,
    Int32Placeholder,
    Int32Variable,
    Int32Constant
)

from .int64 import (
    int64factory,
    Int64Tensor,
    Int64Placeholder,
    Int64Variable,
    Int64Constant
)

from .fixed import (
    _validate_fixedpoint_config,
    fixed100_ni,
    fixed100_i,
    fixed64_ni,
    fixed64_i
)

_validate_fixedpoint_config(fixed100_ni, int100factory)
_validate_fixedpoint_config(fixed100_i, int100factory)
_validate_fixedpoint_config(fixed64_ni, int64factory)
_validate_fixedpoint_config(fixed64_i, int64factory)

__all__ = [
    'Int100Constant',
    'Int100Placeholder',
    'Int100Variable',
    'Int100Tensor',
    'PrimeTensor',
    'PrimePlaceholder',
    'PrimeVariable',
    'PrimeConstant',
    'Int32Tensor',
    'Int32Placeholder',
    'Int32Variable',
    'Int32Constant',
    'Int64Tensor',
    'Int64Placeholder',
    'Int64Variable',
    'Int64Constant',
]
